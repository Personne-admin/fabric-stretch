from __future__ import annotations

from dataclasses import dataclass, field, fields

from .mechanics import normalize_motion

#: Exposures at or above this value are reported to clip on the rig's sensor.
EXPOSURE_WARN_MS = 20.0
#: Largest accepted relative difference between a requested and the exposure the camera really uses
#: (colour calibration charts, measurement runs).
EXPOSURE_TOLERANCE = 0.02
#: Default distance the firmware moves back after an overload trip (command steps).
DEFAULT_RELEASE_STEPS = 200

DIRECTIONS = ("warp", "weft", "bias")
MOTION_MODES = ("loading", "release")
CAPTURE_MODES = ("rgb", "sequential")
#: Setup fields that a preset may leave out; loading such a preset keeps the current choice.
OPTIONAL_PRESET_KEYS = (
    "capture_mode",
    "polarization",
    "include_diffuse",
    "release_steps",
)


@dataclass
class MeasurementSettings:
    count: int = 36
    step_size: int = 100  # command steps per increment, always >= 0 (direction is ``motion_mode``)
    motion_mode: str = "loading"
    led_mask: int = 85  # RGB mode only
    led_power: int = 128
    settle_ms: int = 50
    cam_exposure_ms: float = 15.0
    sample_id: str = "fabric_001"
    direction: str = "warp"
    clamp_width_m: float = 0.20
    max_stretch_mm: float = 0.0  # 0 = off
    max_force_n: float = 0.0  # 0 = off
    speed_mult: int = 1
    relax_enabled: bool = False
    relax_indices: list[int] = field(default_factory=list)
    relax_duration_ms: int = 5000
    capture_mode: str = "rgb"
    polarization: bool = False  # sequential mode: also capture the B LED set
    include_diffuse: bool = False  # sequential mode: add a diffuse reference shot
    release_steps: int = DEFAULT_RELEASE_STEPS  # firmware auto-release after an overload trip

    @classmethod
    def from_preset(cls, preset: dict, current: MeasurementSettings | None = None) -> MeasurementSettings:
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in preset.items() if k in known}
        if current is not None:
            for key in OPTIONAL_PRESET_KEYS:
                values.setdefault(key, getattr(current, key))
        legacy_steps = preset.get(
            "commanded_motor_steps_per_increment",
            preset.get("step_size", cls.step_size),
        )
        values["step_size"], legacy_mode = normalize_motion(legacy_steps, preset.get("motion_mode"))
        values["motion_mode"] = legacy_mode
        values["relax_indices"] = sorted(int(i) for i in preset.get("relax_indices", []))

        return cls(**values)

    def to_preset(self, name: str | None = None) -> dict:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["relax_indices"] = sorted(self.relax_indices)
        if name is not None:
            data["name"] = name

        return data

    def validate(self) -> None:
        problems = []
        if self.count < 1:
            problems.append("Captures must be at least 1.")
        if self.step_size < 0:
            problems.append("Step count must not be negative.")
        if not 0 <= self.led_mask <= 255:
            problems.append("LED mask must be 0..255.")
        if not 0 <= self.led_power <= 255:
            problems.append("LED power must be 0..255.")
        if self.settle_ms < 0:
            problems.append("Settle time must not be negative.")
        if self.cam_exposure_ms <= 0:
            problems.append("Camera exposure must be positive.")
        if self.clamp_width_m <= 0:
            problems.append("Clamp width must be positive.")
        if self.max_stretch_mm < 0 or self.max_force_n < 0:
            problems.append("Max stretch and max force must not be negative (0 = off).")
        if self.speed_mult < 1:
            problems.append("Speed multiplier must be at least 1.")
        if self.release_steps < 0:
            problems.append("Release steps must not be negative.")
        if self.direction not in DIRECTIONS:
            problems.append(f"Direction must be one of {', '.join(DIRECTIONS)}.")
        if self.motion_mode not in MOTION_MODES:
            problems.append(f"Motion must be one of {', '.join(MOTION_MODES)}.")
        if self.capture_mode not in CAPTURE_MODES:
            problems.append(f"Capture mode must be one of {', '.join(CAPTURE_MODES)}.")
        if not self.sample_id.strip() or any(c in self.sample_id for c in "/\\:"):
            problems.append("Sample ID must be a plain folder name.")
        if problems:
            raise ValueError("\n".join(problems))
