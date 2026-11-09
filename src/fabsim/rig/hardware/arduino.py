from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator

from . import protocol

log = logging.getLogger(__name__)


class ControllerError(RuntimeError):
    pass


class ArduinoController:
    def __init__(self, port_factory: Callable[..., object] | None = None):
        self._port_factory = port_factory
        self.port = None
        self.simulated = False
        self._tx_lock = threading.Lock()
        self._conn_lock = threading.RLock()
        self._rx_buf = ""
        self._listeners: list[Callable[[str], None]] = []
        self._listeners_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stop_reader = threading.Event()

        self.move_done = threading.Event()
        self.move_aborted = threading.Event()
        self.stopped = threading.Event()
        self.last_move_abort: dict = {}
        self.last_stop_remaining: int | None = None
        self.lights_on = threading.Event()
        self.power_ack = threading.Event()
        self.limit_armed = threading.Event()
        self.limit_tripped = threading.Event()
        self.last_trip_force: float | None = None
        self._tensions: queue.Queue[str] = queue.Queue()

    @staticmethod
    def list_ports() -> list[str]:
        import serial.tools.list_ports

        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port_name: str, baud: int = protocol.BAUD, settle_s: float = 2.0) -> None:
        with self._conn_lock:
            self.disconnect()
            factory = self._port_factory
            if factory is None:
                import serial

                factory = serial.Serial
            self.port = factory(port_name, baud, timeout=0.1, write_timeout=2.0)
            time.sleep(settle_s)
            self.port.reset_input_buffer()
            self._rx_buf = ""
            self._stop_reader.clear()
            self._reader = threading.Thread(target=self._read_loop, name="serial-reader", daemon=True)
            self._reader.start()
        log.info("Connected to %s", port_name)

    def disconnect(self) -> None:
        with self._conn_lock:
            self._stop_reader.set()
            reader, self._reader = self._reader, None
            if reader is not None and reader is not threading.current_thread():
                reader.join(timeout=1.0)
            port, self.port = self.port, None
            if port is not None and port.is_open:
                port.close()

    @property
    def connected(self) -> bool:
        return self.port is not None and self.port.is_open

    def _read_loop(self) -> None:
        while not self._stop_reader.is_set():
            port = self.port
            if port is None:
                return
            try:
                data = port.read(256)
            except Exception as exc:  # port unplugged or closed under us
                log.error("Serial read failed: %s", exc)

                return
            if data:
                self.feed(data)
            else:
                time.sleep(0.005)

    def add_listener(self, callback: Callable[[str], None]) -> None:
        with self._listeners_lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def feed(self, data: bytes) -> list[str]:
        lines = []
        self._rx_buf += data.decode(errors="replace")
        while "\n" in self._rx_buf:
            line, self._rx_buf = self._rx_buf.split("\n", 1)
            line = line.strip()
            if line:
                lines.append(line)
                self._handle_line(line)

        return lines

    def _handle_line(self, line: str) -> None:
        reply = protocol.parse_reply(line)
        if reply.kind == protocol.MOVE_DONE:
            self.move_done.set()
        elif reply.kind == protocol.MOVE_ABORTED:
            self.last_move_abort = reply.data
            self.move_aborted.set()
        elif reply.kind == protocol.STOPPED:
            self.last_stop_remaining = reply.data.get("remaining")
            self.stopped.set()
        elif reply.kind == protocol.LIMIT_TRIP:
            self.last_trip_force = reply.data.get("force_n")
            self.limit_tripped.set()
        elif reply.kind == protocol.LEDS_OK:
            self.lights_on.set()
        elif reply.kind == protocol.POWER:
            self.power_ack.set()
        elif reply.kind == protocol.LIMIT_ARMED:
            self.limit_armed.set()
        elif reply.kind == protocol.TENSION:
            self._tensions.put(line)
        with self._listeners_lock:
            listeners = list(self._listeners)
        for callback in listeners:
            try:
                callback(line)
            except Exception:
                log.exception("Listener failed")

    def send(self, cmd: str) -> None:
        if not self.connected:
            raise ControllerError("Not connected")
        with self._tx_lock:
            self.port.write((cmd + "\n").encode())
        log.debug("TX: %s", cmd)

    def jog(self, steps: int) -> None:
        self.move_done.clear()
        self.move_aborted.clear()
        self.last_move_abort = {}
        self.send(protocol.jog(steps))

    def stop(self) -> None:
        self.stopped.clear()
        self.last_stop_remaining = None
        self.send(protocol.stop())

    def wait_stopped(self, timeout: float = 1.0) -> bool:
        return self.stopped.wait(timeout)

    def set_leds(self, mask: int) -> None:
        self.lights_on.clear()
        self.send(protocol.leds(mask))

    def set_power(self, value: int) -> None:
        self.power_ack.clear()
        self.send(protocol.power(value))

    def set_speed(self, mult: int) -> None:
        self.send(protocol.speed(mult))

    def tare_tension(self) -> None:
        self.send(protocol.tare())

    def read_tension(self) -> None:
        self.send(protocol.read_tension())

    def request_tension(self, timeout: float = 2.0) -> str | None:
        while True:
            try:
                self._tensions.get_nowait()
            except queue.Empty:
                break
        self.send(protocol.read_tension())
        try:
            return self._tensions.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_force_limit(
        self, force_n: float, release_steps: int = 0, release_dir_positive: bool = False
    ) -> None:
        self.limit_armed.clear()
        if force_n > 0:
            self.clear_limit_trip()
        self.send(protocol.force_limit(force_n, release_steps, release_dir_positive))

    def clear_limit_trip(self) -> None:
        self.limit_tripped.clear()
        self.last_trip_force = None

    def wait_move_done(self, timeout: float = 30.0) -> bool:
        return self.move_done.wait(timeout)

    def wait_lights_on(self, timeout: float = 2.0) -> bool:
        return self.lights_on.wait(timeout)

    def wait_power_ack(self, timeout: float = 0.5) -> bool:
        return self.power_ack.wait(timeout)

    def wait_limit_armed(self, timeout: float = 1.0) -> bool:
        return self.limit_armed.wait(timeout)

    def lights_off(self) -> None:
        self.set_leds(0)
        self.set_power(0)

    @contextlib.contextmanager
    def lights(self, mask: int, power: int, log_fn: Callable[[str], None] | None = None) -> Iterator[None]:
        try:
            self.set_power(power)
            if not self.wait_power_ack(0.5):
                time.sleep(0.05)
            if mask != 0:
                self.set_leds(mask)
                if not self.wait_lights_on(2.0):
                    if log_fn:
                        log_fn("  WARNING: LEDs not confirmed, retrying")
                    self.set_leds(0)
                    time.sleep(0.05)
                    self.set_power(power)
                    self.wait_power_ack(0.5)
                    self.set_leds(mask)
                    if not self.wait_lights_on(2.0):
                        raise ControllerError(
                            f"LED mask {mask} never confirmed; aborting frame to avoid a dark capture"
                        )
            else:
                self.set_leds(0)
            yield
        finally:
            if self.connected:
                with contextlib.suppress(Exception):
                    self.lights_off()
