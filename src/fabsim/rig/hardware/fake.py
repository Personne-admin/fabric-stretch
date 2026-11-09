from __future__ import annotations

import re
import threading

#: Pseudo port name offered by the GUI to run without hardware.
SIMULATED_PORT = "SIMULATED"


class FakePort:
    def __init__(
        self,
        name: str = SIMULATED_PORT,
        baud: int = 9600,
        timeout: float = 0.1,
        write_timeout: float | None = None,
        *,
        force_per_step: float = 0.0005,
        confirm_leds: bool = True,
        answer_tension: bool = True,
        answer_moves: bool = True,
        answer_limit: bool = True,
        firmware_trips: bool = True,
    ):
        self.name = name
        self.is_open = True
        self.force_per_step = force_per_step
        self.confirm_leds = confirm_leds
        self.answer_tension = answer_tension
        self.answer_moves = answer_moves
        self.answer_limit = answer_limit
        self.firmware_trips = firmware_trips
        self.position = 0
        self.led_mask = 0
        self.power = 0
        self.limit_n = 0.0
        self.log: list[str] = []
        self._timeout = timeout
        self._out = bytearray()
        self._cv = threading.Condition()
        self._in = ""

    @property
    def in_waiting(self) -> int:
        return len(self._out)

    def reset_input_buffer(self) -> None:
        with self._cv:
            self._out.clear()

    def close(self) -> None:
        self.is_open = False

    def read(self, size: int = 1) -> bytes:
        with self._cv:
            if not self._out:
                self._cv.wait(self._timeout)
            data = bytes(self._out[:size])
            del self._out[:size]

            return data

    def write(self, data: bytes) -> int:
        self._in += data.decode()
        while "\n" in self._in:
            command, self._in = self._in.split("\n", 1)
            self.log.append(command)
            for line in self._answer(command.strip()):
                self._emit(line)

        return len(data)

    def _emit(self, line: str) -> None:
        with self._cv:
            self._out += (line + "\n").encode()
            self._cv.notify_all()

    def force_n(self) -> float:
        return self.position * self.force_per_step

    def _answer(self, cmd: str) -> list[str]:
        if not cmd:
            return []
        head, rest = cmd[0], cmd[1:]
        if head in "+-":
            steps = int(rest or 0)
            self.position += steps if head == "+" else -steps
            replies = [f"jog: {head}{steps}"]
            if self.firmware_trips and self.limit_n > 0 and abs(self.force_n()) >= self.limit_n:
                replies.append(self._telemetry("LIMIT_TRIP") + f" limit_n={self.limit_n:.6f}")
            elif steps and self.answer_moves:
                replies.append("MOVE_DONE")

            return replies
        if head == "L":
            self.led_mask = int(rest or 0)
            if self.led_mask == 0:
                return ["leds: 0"]

            return [f"leds: {self.led_mask} OK"] if self.confirm_leds else []
        if head == "P":
            self.power = int(rest or 0)

            return [f"power: {self.power}"]
        if head == "t":
            return [self._telemetry("TENSION")] if self.answer_tension else []
        if head == "F":
            values = re.findall(r"[-+0-9.]+", rest)
            self.limit_n = float(values[0]) if values else 0.0
            if self.limit_n <= 0:
                return ["limit: off"]
            if not self.answer_limit:
                return []

            return [
                f"LIMIT_ARMED limit_n={self.limit_n:.6f}"
                f" release_motor_steps={values[1] if len(values) > 1 else 0}"
                f" release_dir_positive={values[2] if len(values) > 2 else 0}"
            ]
        if head == "i":
            return ["status: API ready"]
        if head == "@":
            return ["zero: 0"]
        if head == "#":
            return ["trigger"]
        if head in "V*":
            return []

        return [f"?{head}"]

    def _telemetry(self, prefix: str) -> str:
        counts = -round(self.force_n() / 0.0125)

        return (
            f"{prefix} force_n={self.force_n():.6f} raw_counts={counts} zero_counts=0 "
            f"calibration_n_per_count=0.01250000"
        )
