from __future__ import annotations

import pytest

from fabsim.rig import mechanics
from fabsim.rig.mechanics import DEFAULT_MOTOR_MECHANICS, MotorMechanics


def test_default_mechanics_match_the_recorded_datasets():
    m = DEFAULT_MOTOR_MECHANICS
    assert m.motor_steps_per_output_rev == 9800
    assert m.linear_um_per_motor_step == pytest.approx(0.15306, abs=1e-5)
    # 8000 steps per increment are 1.224 mm in every recorded session.
    assert m.displacement_mm(8000) == pytest.approx(1.2245, abs=1e-3)
    assert m.displacement_mm(-8000) == m.displacement_mm(8000)


def test_microsteps_scale_the_travel():
    finer = MotorMechanics(microsteps=2)
    assert finer.linear_um_per_motor_step == pytest.approx(
        DEFAULT_MOTOR_MECHANICS.linear_um_per_motor_step / 2
    )


def test_steps_for_mm_round_trips():
    m = DEFAULT_MOTOR_MECHANICS
    assert m.steps_for_mm(1.0) == pytest.approx(6533, abs=1)
    assert m.displacement_mm(m.steps_for_mm(2.5)) == pytest.approx(2.5, abs=1e-3)


def test_session_fields_document_the_driver_assumption():
    fields = DEFAULT_MOTOR_MECHANICS.as_session_fields()
    assert fields["microsteps"] == 1
    assert fields["driver_pulses_per_command_step"] == 16
    assert fields["motor_steps_per_output_rev"] == 9800


def test_normalize_motion_handles_legacy_signed_counts():
    assert mechanics.normalize_motion(-300) == (300, "release")
    assert mechanics.normalize_motion(300) == (300, "loading")
    assert mechanics.normalize_motion(300, "release") == (300, "release")
    assert mechanics.normalize_motion(-300, "loading") == (300, "loading")
    assert mechanics.normalize_motion(5, "bogus") == (5, "loading")


def test_signed_motor_steps():
    assert mechanics.signed_motor_steps(100, "loading") == 100
    assert mechanics.signed_motor_steps(100, "release") == -100
    assert mechanics.signed_motor_steps(-100, "release") == -100


def test_move_duration_follows_the_firmware_timer():
    # 32 ticks of 32 us per command step at speed 1; a move of N runs N + 1 periods.
    assert mechanics.tick_period_us(1) == pytest.approx(32.0)
    assert mechanics.tick_period_us(2) == pytest.approx(16.0)
    assert mechanics.move_duration_s(0, 1) == pytest.approx(32 * 32e-6)
    assert mechanics.move_duration_s(8000, 1) == pytest.approx(8001 * 1.024e-3)
    assert mechanics.move_duration_s(-8000, 2) == pytest.approx(8001 * 0.512e-3)
    assert mechanics.tick_period_us(1000) == pytest.approx(1.0)  # the firmware clamps at 2 counts


def test_every_move_travels_one_command_step_more_than_commanded():
    # firmware/README.md#extra-motor-period: a move of N steps runs N + 1 periods
    assert mechanics.travelled_steps(100, 0) == 0
    assert mechanics.travelled_steps(100, 3) == 303
    assert mechanics.travelled_steps(-100, 2) == 202
    assert mechanics.travelled_steps(0, 5) == 0  # no commanded step, no move
