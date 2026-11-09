from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from . import session
from .colorcorrect import ColorCorrector
from .hardware import protocol
from .hardware.arduino import ArduinoController
from .hardware.camera import frame_to_pil
from .mechanics import (
    DEFAULT_MOTOR_MECHANICS,
    MotorMechanics,
    move_duration_s,
    signed_motor_steps,
    travelled_steps,
)
from .settings import EXPOSURE_TOLERANCE, MeasurementSettings

log = logging.getLogger(__name__)

#: A shot darker than this fraction of the running mean of its role is reported as missed illumination.
MISSED_ILLUMINATION_RATIO = 0.4
#: Safety factor and constant slack added to the computed duration of a move.
MOVE_TIMEOUT_FACTOR = 1.5
MOVE_TIMEOUT_SLACK_S = 2.0
#: Seconds to wait for ``LIMIT_ARMED`` after arming the firmware force limit.
LIMIT_ARM_TIMEOUT_S = 1.0
#: Seconds to wait for the load-cell answer to ``t``.
TENSION_TIMEOUT_S = 2.0
#: Pause between the shots of one state.
INTER_SHOT_S = 0.05

DONE_COMPLETE = "complete"
DONE_ABORTED = "aborted"
DONE_ERROR = "error"


class RunAbort(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CaptureError(RuntimeError):
    pass


class AutomaticDispatcher:
    def __init__(
        self,
        ctrl: ArduinoController,
        cam,
        run_dir: Path,
        settings: MeasurementSettings,
        plan: session.CapturePlan,
        led_config: dict,
        color_bundle=None,
        keep_raw: bool = True,
        mechanics: MotorMechanics = DEFAULT_MOTOR_MECHANICS,
        on_image: Callable[[Image.Image], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        on_done: Callable[[str, str | None], None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ):
        self.ctrl = ctrl
        self.cam = cam
        self.run_dir = Path(run_dir)
        self.settings = settings
        self.plan = plan
        self.led_config = led_config
        self.mechanics = mechanics
        self.on_image = on_image
        self.on_progress = on_progress
        self.on_done = on_done
        self.on_log = on_log
        self._color_bundle = color_bundle
        self._keep_raw = keep_raw
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._motion_lock = threading.Lock()
        self._stop_requested = False
        self._abort_reason: str | None = None
        self._role_brightness: dict[str, float] = {}
        self._limit_armed = False
        self.writer: session.SessionWriter | None = None
        self.corrector: ColorCorrector | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def cancel(self, reason: str | None = None) -> None:
        if reason and self._abort_reason is None:
            self._abort_reason = reason
        self._cancel.set()
        with self._motion_lock:
            if not self._stop_requested and self.ctrl.connected:
                try:
                    self.ctrl.stop()
                    self._stop_requested = True
                except Exception as exc:
                    self._log(f"WARNING: could not send motor stop: {exc}")

    def start(self) -> None:
        if self.running:
            return
        self.prepare()
        self._thread = threading.Thread(target=self.run, name="measurement", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def prepare(self) -> None:
        s = self.settings
        s.validate()
        if not self.cam.is_open:
            raise RuntimeError("The camera is not open.")
        if not self.ctrl.connected:
            raise RuntimeError("The controller is not connected.")
        self._cancel.clear()
        self._stop_requested = False
        self._abort_reason = None
        actual_ms = self.cam.set_exposure(s.cam_exposure_ms)
        if abs(actual_ms - s.cam_exposure_ms) > EXPOSURE_TOLERANCE * s.cam_exposure_ms:
            raise RuntimeError(
                f"The camera uses an exposure of {actual_ms:g} ms instead of the requested "
                f"{s.cam_exposure_ms:g} ms; choose another exposure."
            )
        camera_info = {
            **self.cam.info(),
            "exposure_ms_requested": s.cam_exposure_ms,
            "exposure_ms_actual": actual_ms,
        }
        simulated = bool(getattr(self.cam, "simulated", False) or getattr(self.ctrl, "simulated", False))
        self.corrector = ColorCorrector(
            self._color_bundle, actual_ms, self._keep_raw, self.run_dir, self._log
        )
        test_id = f"{s.direction}_stretch_{time.strftime('%Y%m%d_%H%M%S')}"
        meta = session.build_session_meta(
            s,
            test_id=test_id,
            plan=self.plan,
            led_config=self.led_config,
            camera_info=camera_info,
            color_stamp=self.corrector.stamp,
            simulated=simulated,
            mechanics=self.mechanics,
        )
        self.writer = session.SessionWriter(self.run_dir, meta, self.mechanics)
        self.writer.start()
        self.corrector.save_bundle_copy()

    def _log(self, msg: str) -> None:
        log.info(msg)
        if self.on_log:
            self.on_log(msg)

    def _sleep(self, duration_s: float, poll_interval: float = 0.1) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline and not self._cancel.is_set():
            time.sleep(max(0.0, min(poll_interval, deadline - time.monotonic())))

    def run(self) -> None:
        status = DONE_COMPLETE
        try:
            self._measure()
            if self._cancel.is_set():
                status = DONE_ABORTED
        except RunAbort as abort:
            self.cancel(abort.reason)
            self._log(f"ABORT: {abort.reason}")
            status = DONE_ABORTED
        except Exception as exc:
            log.exception("Measurement failed")
            self.cancel(f"error: {exc}")
            self._log(f"ERROR: {exc}")
            status = DONE_ERROR
        finally:
            self._cleanup_hardware()
            try:
                self.writer.finish(self._abort_reason, status)
            except Exception as exc:
                log.exception("Could not finalise session.json")
                self._log(f"ERROR: could not finalise session.json: {exc}")
                self.cancel(f"error: could not finalise session.json ({exc})")
                status = DONE_ERROR
        if status == DONE_COMPLETE:
            self._log("Auto measurement complete")
        elif self._abort_reason:
            self._log(f"Auto measurement aborted: {self._abort_reason}")
        else:
            self._log("Auto measurement cancelled")
        if self.on_done:
            self.on_done(status, self._abort_reason)

    def _cleanup_hardware(self) -> None:
        self.ctrl.clear_limit_trip()
        if not self.ctrl.connected:
            return
        if self._stop_requested and not self.ctrl.wait_stopped(1.0):
            self._log("WARNING: motor stop not acknowledged; check the controller and use the panic button")
        if self._limit_armed:
            with contextlib.suppress(Exception):
                self.ctrl.set_force_limit(0)
        with contextlib.suppress(Exception):
            self.ctrl.lights_off()

    def _measure(self) -> None:
        s = self.settings
        signed = signed_motor_steps(s.step_size, s.motion_mode)
        self._log(
            f"Auto measurement start ({s.direction} {s.motion_mode}, {s.step_size} motor steps/increment)"
        )
        self.ctrl.clear_limit_trip()  # a trip of an earlier run must not abort this one
        self.ctrl.set_speed(s.speed_mult)
        self.ctrl.set_power(0)
        self.ctrl.set_leds(0)
        if s.max_force_n > 0:
            self._arm_force_limit(signed)

        for i in range(s.count):
            if self._cancel.is_set():
                return
            displacement_mm = round(self.mechanics.displacement_mm(travelled_steps(s.step_size, i)), 6)
            if s.max_stretch_mm > 0 and displacement_mm > s.max_stretch_mm:
                raise RunAbort(f"stretch_limit_exceeded ({displacement_mm:g} mm > {s.max_stretch_mm:g} mm)")

            self._log(f"Capture {i + 1}/{s.count}")
            if s.step_size != 0 and i > 0 and not self._jog_and_wait(signed):
                return
            self._sleep(s.settle_ms / 1000.0)
            if self._cancel.is_set():
                return
            self._check_firmware_trip()

            record = self._capture_state(i, displacement_mm, str(i), is_relaxation_hold=False)
            if record is None or self._check_force_safety(record):
                return

            if s.relax_enabled and i in s.relax_indices:
                self._log(f"  Relax pause at {i}: {s.relax_duration_ms} ms")
                self._sleep(s.relax_duration_ms / 1000.0)
                if self._cancel.is_set():
                    return
                record = self._capture_state(i, displacement_mm, f"{i}_R", is_relaxation_hold=True)
                if record is None or self._check_force_safety(record):
                    return
            if self.on_progress:
                self.on_progress(i + 1, s.count)

    def _arm_force_limit(self, signed_steps: int) -> None:
        s = self.settings
        release = s.release_steps if s.step_size != 0 else 0
        with self._motion_lock:
            if self._cancel.is_set():
                return
            self.ctrl.set_force_limit(s.max_force_n, release, release_dir_positive=signed_steps < 0)
            self._limit_armed = True
        if not self.ctrl.wait_limit_armed(LIMIT_ARM_TIMEOUT_S):
            raise RunAbort("force_limit_not_armed (no LIMIT_ARMED reply from the controller)")

    def _jog_and_wait(self, signed_steps: int) -> bool:
        s = self.settings
        with self._motion_lock:
            if self._cancel.is_set():
                return False
            self.ctrl.jog(signed_steps)
        timeout = move_duration_s(signed_steps, s.speed_mult) * MOVE_TIMEOUT_FACTOR + MOVE_TIMEOUT_SLACK_S
        deadline = time.monotonic() + timeout
        while not self.ctrl.wait_move_done(0.05):
            if self._cancel.is_set():
                return False
            if self.ctrl.limit_tripped.is_set():
                return True  # the firmware stops without MOVE_DONE; the caller reports the trip
            if self.ctrl.move_aborted.is_set():
                reason = self.ctrl.last_move_abort.get("reason", "unknown")
                if reason != "force_limit":
                    raise RunAbort(f"move_aborted ({reason})")
            if time.monotonic() > deadline:
                raise RunAbort(f"move_timeout (no MOVE_DONE within {timeout:.1f} s)")

        if self._cancel.is_set():
            return False
        if self.ctrl.move_aborted.is_set():
            reason = self.ctrl.last_move_abort.get("reason", "unknown")
            if reason != "force_limit":
                raise RunAbort(f"move_aborted ({reason})")
        return True

    def _check_firmware_trip(self) -> None:
        if self.ctrl.limit_tripped.is_set():
            force = self.ctrl.last_trip_force
            raise RunAbort(f"firmware_force_limit ({force} N)")

    def _check_force_safety(self, force_record: dict) -> bool:
        limit = self.settings.max_force_n
        force_n = force_record.get("force_n")
        reason = protocol.overload_reason(force_n, limit)
        self._log(
            f"  overload check: raw={force_record.get('raw_counts')} counts force={force_n} N "
            f"limit={limit:.3f} N result={reason or 'within_limit'}"
        )
        if reason:
            self._log(f"SAFETY ABORT: |force| {force_n:.3f} N >= max {limit:.3f} N")
            self.cancel(reason)

            return True

        return False

    def _capture_state(
        self,
        step_index: int,
        displacement_mm: float,
        label: str,
        is_relaxation_hold: bool,
    ) -> dict | None:
        s = self.settings
        frame_id = len(self.writer.frames)
        image_files: dict[str, str] = {}
        for role, mask in self.plan:
            if self._cancel.is_set():
                return None
            fname = f"img_{frame_id:04d}_{role}.png"
            image = self._capture_role(mask, role, fname)
            image_files[f"image_{role}"] = fname
            if self.on_image:
                self.on_image(image)
            self._sleep(INTER_SHOT_S)

        force_index = len(self.writer.forces)
        force_record = self._read_force(force_index, label)
        self.writer.add_frame(
            session.frame_record(
                frame_id,
                step_index,
                displacement_mm,
                force_index,
                session.line_load_n_per_m(force_record.get("force_n"), s.clamp_width_m),
                image_files,
                s.settle_ms,
                is_relaxation_hold,
            ),
            force_record,
        )

        return force_record

    def _capture_role(self, mask: int, role: str, fname: str) -> Image.Image:
        s = self.settings
        with self.ctrl.lights(mask, s.led_power, self._log):
            frame = self.cam.capture_frame()
        if frame is None:
            raise CaptureError(f"camera returned no frame for role {role!r}")
        image = frame_to_pil(frame)
        if mask != 0 and not getattr(self.cam, "simulated", False):
            self._check_brightness(image, role, fname)

        out_path = self.writer.images_dir / fname
        corrected = self.corrector.correct(image, role)
        if corrected is None:
            image.save(out_path)
        else:
            if self.corrector.keep_raw:
                raw_dir = self.run_dir / "images_raw"
                raw_dir.mkdir(exist_ok=True)
                image.save(raw_dir / fname)
            corrected.save(out_path)
            image = corrected
        self._log(f"  saved {fname}")

        return image

    def _check_brightness(self, image: Image.Image, role: str, fname: str) -> None:
        mean8 = float(np.asarray(image).mean())
        median = self._role_brightness.get(role)
        if median is not None and mean8 < MISSED_ILLUMINATION_RATIO * median:
            self._log(
                f"  WARNING: {fname} mean {mean8:.1f} << running median {median:.1f}, "
                "possible missed illumination"
            )
        self._role_brightness[role] = mean8 if median is None else 0.9 * median + 0.1 * mean8

    def _read_force(self, index: int, label: str) -> dict:
        record = session.blank_force_record(index, label)
        line = self.ctrl.request_tension(timeout=TENSION_TIMEOUT_S)
        if line is None:
            self._log("  WARNING: tension timed out")

            return record
        record["raw"] = line
        telemetry = protocol.parse_force_telemetry(line) if "BUSY" not in line and "N/A" not in line else None
        if telemetry is None:
            record["status"] = "invalid"
            self._log(f"  WARNING: controller returned {line!r}")

            return record
        record.update(telemetry)
        record["value"] = telemetry["force_n"]
        record["status"] = "ok"
        self._log(
            f"  force telemetry: raw={telemetry['raw_counts']} counts zero={telemetry['zero_counts']} counts "
            f"coef={telemetry['calibration_n_per_count']} N/count -> {telemetry['force_n']:.6f} N"
        )

        return record
