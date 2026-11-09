from __future__ import annotations

from dataclasses import dataclass

#: STEP pin pulses per command step (``Constants::Timer::Fraction`` / 2 toggles per pulse).
DRIVER_PULSES_PER_COMMAND_STEP = 16
#: Period of the firmware step timer at speed multiplier 1 (OCR2A = 63, prescaler 8, 16 MHz).
BASE_TICK_US = 32.0
#: Timer ticks per command step (``Constants::Timer::Fraction``).
TICKS_PER_COMMAND_STEP = 32
#: Smallest timer period in counts that the firmware allows (see ``Stepper::set_speed_multiplier``).
_MIN_TIMER_COUNTS = 2
_BASE_TIMER_COUNTS = 64


@dataclass(frozen=True)
class MotorMechanics:

    screw_pitch_um_per_output_rev: float = 1500.0
    gear_ratio: float = 49.0
    motor_full_steps_per_rev: int = 200
    microsteps: int = 1

    @property
    def motor_steps_per_output_rev(self) -> int:
        return self.motor_full_steps_per_rev * int(self.gear_ratio) * self.microsteps

    @property
    def linear_um_per_motor_step(self) -> float:
        return self.screw_pitch_um_per_output_rev / self.motor_steps_per_output_rev

    def displacement_mm(self, motor_steps: int) -> float:
        return abs(motor_steps) * self.linear_um_per_motor_step / 1000.0

    def steps_for_mm(self, millimetres: float) -> int:
        return round(abs(millimetres) * 1000.0 / self.linear_um_per_motor_step)

    def as_session_fields(self) -> dict:
        return {
            "screw_pitch_um_per_output_rev": self.screw_pitch_um_per_output_rev,
            "gear_ratio": self.gear_ratio,
            "motor_full_steps_per_rev": self.motor_full_steps_per_rev,
            "microsteps": self.microsteps,
            "driver_pulses_per_command_step": DRIVER_PULSES_PER_COMMAND_STEP,
            "motor_steps_per_output_rev": self.motor_steps_per_output_rev,
            "linear_um_per_motor_step": self.linear_um_per_motor_step,
        }


DEFAULT_MOTOR_MECHANICS = MotorMechanics()


def normalize_motion(
    step_count: int, motion_mode: str | None = None
) -> tuple[int, str]:
    legacy_mode = "release" if step_count < 0 else "loading"
    mode = motion_mode if motion_mode in {"loading", "release"} else legacy_mode

    return abs(int(step_count)), mode


def signed_motor_steps(step_count: int, motion_mode: str) -> int:
    magnitude, mode = normalize_motion(step_count, motion_mode)

    return magnitude if mode == "loading" else -magnitude


def travelled_steps(step_size: int, moves: int) -> int:
    return moves * (abs(int(step_size)) + 1) if step_size else 0


def tick_period_us(speed_mult: int = 1) -> float:
    counts = max(_BASE_TIMER_COUNTS // max(int(speed_mult), 1), _MIN_TIMER_COUNTS)

    return counts * BASE_TICK_US / _BASE_TIMER_COUNTS


def move_duration_s(steps: int, speed_mult: int = 1) -> float:
    return (
        (abs(int(steps)) + 1)
        * TICKS_PER_COMMAND_STEP
        * tick_period_us(speed_mult)
        * 1e-6
    )
