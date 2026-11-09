from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .. import __version__
from .mechanics import (
    DEFAULT_MOTOR_MECHANICS,
    MotorMechanics,
    signed_motor_steps,
    travelled_steps,
)
from .settings import MeasurementSettings

SCHEMA_VERSION = 2

CapturePlan = list[tuple[str, int]]


class CapturePlanError(ValueError):
    pass


def build_capture_plan(settings: MeasurementSettings, led_config: dict) -> CapturePlan:
    if settings.capture_mode == "rgb":
        return [("rgb", settings.led_mask)]

    set_a = list(led_config.get("pol_a_bits", []))
    if len(set_a) != 3:
        raise CapturePlanError(
            f"Set A needs exactly three LEDs (A1-A3), {len(set_a)} are mapped; use Identify LEDs first."
        )
    azimuths = led_config.get("light_azimuths_deg")
    if not isinstance(azimuths, list) or len(azimuths) != 3 or led_config.get("light_elevation_deg") is None:
        raise CapturePlanError(
            "The light directions of set A are not calibrated; run Calibrate Light Angles first."
        )
    set_b = list(led_config.get("pol_b_bits", []))
    if settings.polarization and len(set_b) != 3:
        raise CapturePlanError(
            f"Set B needs exactly three LEDs (B1-B3), {len(set_b)} are mapped; use Identify LEDs first."
        )

    plan: CapturePlan = []
    if settings.include_diffuse:
        mask = int(led_config.get("diffuse_mask", 0))
        if mask == 0:
            raise CapturePlanError("No diffuse LED is mapped; use Identify LEDs first.")
        plan.append(("diffuse", mask))
    plan += [(f"ps_{i}", 1 << bit) for i, bit in enumerate(set_a, start=1)]
    if settings.polarization:
        plan += [(f"ps_b_{i}", 1 << bit) for i, bit in enumerate(set_b, start=1)]

    return plan


def plan_to_json(plan: CapturePlan) -> list[dict]:
    return [{"role": role, "mask": mask} for role, mask in plan]


def run_dir_in_use(path: Path) -> bool:
    return any((path / name).exists() for name in ("session.json", "frames.json", "force.json"))


def next_free_run_dir(base: Path) -> Path:
    if not run_dir_in_use(base):
        return base
    n = 2
    while run_dir_in_use(base.with_name(f"{base.name}_{n}")):
        n += 1

    return base.with_name(f"{base.name}_{n}")


def run_dir_for(project_dir: Path, settings: MeasurementSettings) -> Path:
    return project_dir / settings.sample_id.strip() / f"{settings.direction}_{settings.motion_mode}"


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def software_info() -> dict:
    return {
        "fabsim": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
    }


def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def line_load_n_per_m(force_n: float | None, clamp_width_m: float) -> float | None:
    if force_n is None or clamp_width_m <= 0:
        return None

    return force_n / clamp_width_m


def blank_force_record(index: int, label: str) -> dict:
    return {
        "index": index,
        "label": label,
        "timestamp": local_now(),
        "raw": None,
        "value": None,
        "force_n": None,
        "raw_counts": None,
        "zero_counts": None,
        "calibration_n_per_count": None,
        "status": "timeout",
    }


def frame_record(
    frame_id: int,
    step_index: int,
    displacement_mm: float,
    force_index: int,
    line_load: float | None,
    image_files: dict[str, str],
    settle_ms: int,
    is_relaxation_hold: bool,
) -> dict:
    return {
        "frame_id": frame_id,
        "step_index": step_index,
        "displacement_mm": displacement_mm,
        "force_json_index": force_index,
        "force_csv_row": force_index,  # legacy alias
        "line_load_N_per_m": line_load,
        "settle_ms": settle_ms,
        "is_relaxation_hold": is_relaxation_hold,
        **image_files,
    }


def build_session_meta(
    settings: MeasurementSettings,
    *,
    test_id: str,
    plan: CapturePlan,
    led_config: dict,
    camera_info: dict,
    color_stamp: dict,
    simulated: bool,
    mechanics: MotorMechanics = DEFAULT_MOTOR_MECHANICS,
) -> dict:
    signed = signed_motor_steps(settings.step_size, settings.motion_mode)
    started = local_now()

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": started,
        "started_at": started,
        "finished_at": None,
        "status": "running",
        "simulated": simulated,
        "sample_id": settings.sample_id.strip(),
        "test_id": test_id,
        "direction": settings.direction,
        "branch": settings.motion_mode,
        "motion_mode": settings.motion_mode,
        "count": settings.count,
        "step_size": signed,
        "commanded_step_count": settings.step_size,
        "commanded_motor_steps_per_increment": settings.step_size,
        "requested_displacement_um_per_increment": settings.step_size * mechanics.linear_um_per_motor_step,
        "travelled_displacement_um_per_increment": travelled_steps(settings.step_size, 1)
        * mechanics.linear_um_per_motor_step,
        **mechanics.as_session_fields(),
        "capture_mode": settings.capture_mode,
        "capture_plan": plan_to_json(plan),
        "led_mask": settings.led_mask if settings.capture_mode == "rgb" else None,
        "led_power": settings.led_power,
        "led_config": led_config,
        "settle_ms": settings.settle_ms,
        "cam_exposure_ms": settings.cam_exposure_ms,
        "speed_mult": settings.speed_mult,
        "clamp_width_m": settings.clamp_width_m,
        "max_stretch_mm": settings.max_stretch_mm,
        "max_force_n": settings.max_force_n,
        "force_release_steps": settings.release_steps,
        "relax_enabled": settings.relax_enabled,
        "relax_indices": sorted(settings.relax_indices),
        "relax_duration_ms": settings.relax_duration_ms,
        "abort_reason": None,
        "color_calibration": color_stamp,
        "camera": camera_info,
        "software": software_info(),
        "frames_written": 0,
    }


@dataclass
class SessionWriter:
    run_dir: Path
    meta: dict
    mechanics: MotorMechanics = DEFAULT_MOTOR_MECHANICS
    frames: list[dict] = field(default_factory=list)
    forces: list[dict] = field(default_factory=list)

    @property
    def images_dir(self) -> Path:
        return self.run_dir / "images"

    def start(self) -> None:
        if run_dir_in_use(self.run_dir):
            raise FileExistsError(f"{self.run_dir} already contains a measurement run")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.run_dir / "session.json", self.meta)
        self._write_frames()

    def add_frame(self, frame: dict, force: dict) -> None:
        self.frames.append(frame)
        self.forces.append(force)
        self._write_frames()
        self.meta["frames_written"] = len(self.frames)
        atomic_write_json(self.run_dir / "session.json", self.meta)

    def finish(self, abort_reason: str | None, status: str) -> None:
        self.meta["abort_reason"] = abort_reason
        self.meta["status"] = status
        self.meta["finished_at"] = local_now()
        self.meta["frames_written"] = len(self.frames)
        self._write_frames()
        atomic_write_json(self.run_dir / "session.json", self.meta)

    def _write_frames(self) -> None:
        m = self.meta
        atomic_write_json(
            self.run_dir / "frames.json",
            {
                "sample_id": m["sample_id"],
                "test_id": m["test_id"],
                "direction": m["direction"],
                "branch": m["motion_mode"],
                "motion_mode": m["motion_mode"],
                "clamp_width_m": m["clamp_width_m"],
                "commanded_motor_steps_per_increment": m["commanded_motor_steps_per_increment"],
                "requested_displacement_um_per_increment": m["requested_displacement_um_per_increment"],
                **self.mechanics.as_session_fields(),
                "frames": self.frames,
            },
        )
        atomic_write_json(self.run_dir / "force.json", {"forces": self.forces})
