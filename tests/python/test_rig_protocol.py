from __future__ import annotations

import pytest

from fabsim.rig.hardware import protocol
from fabsim.rig.hardware.arduino import ArduinoController, ControllerError
from fabsim.rig.hardware.fake import FakePort


def test_command_builders():
    assert protocol.jog(200) == "+200"
    assert protocol.jog(-5) == "-5"
    assert protocol.jog(0) == "+0"
    assert protocol.leds(0) == "L0"
    assert protocol.leds(255) == "L255"
    assert protocol.power(128) == "P128"
    assert protocol.speed(2) == "V2"
    assert protocol.speed(0) == "V1"
    assert protocol.tare() == "@"
    assert protocol.read_tension() == "t"
    assert protocol.set_coefficient(0.0125) == "*0.01250000"


@pytest.mark.parametrize("bad", [-1, 256])
def test_leds_and_power_reject_out_of_range(bad):
    with pytest.raises(ValueError):
        protocol.leds(bad)
    with pytest.raises(ValueError):
        protocol.power(bad)


def test_force_limit_is_fixed_point():
    # The firmware's parseFloat cannot read exponent notation.
    assert protocol.force_limit(90.0, 200, True) == "F90.000000,200,1"
    assert protocol.force_limit(1e-5, 0, False) == "F0.000010,0,0"
    assert "e" not in protocol.force_limit(1e-5)
    assert protocol.force_limit(0) == "F0"
    assert protocol.force_limit(-3) == "F0"


def test_parse_force_telemetry_full_line():
    line = "TENSION force_n=-4.312500 raw_counts=345 zero_counts=0 calibration_n_per_count=0.01250000"
    assert protocol.parse_force_telemetry(line) == {
        "force_n": -4.3125,
        "raw_counts": 345,
        "zero_counts": 0,
        "calibration_n_per_count": 0.0125,
    }


def test_parse_force_telemetry_legacy_and_invalid():
    assert protocol.parse_force_telemetry("tension: 1.5 N")["force_n"] == 1.5
    assert protocol.parse_force_telemetry("tension: N/A") is None
    assert protocol.parse_force_telemetry("tension:") is None
    assert protocol.parse_force_telemetry("hello") is None
    assert protocol.parse_force_telemetry("TENSION raw_counts=3") is None


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("MOVE_DONE", protocol.MOVE_DONE),
        ("leds: 5 OK", protocol.LEDS_OK),
        ("leds: 0", protocol.LEDS_OFF),
        ("power: 128", protocol.POWER),
        ("jog: +200", protocol.JOG),
        ("zero: -12", protocol.ZERO),
        ("coef: 0.012500", protocol.COEF),
        ("status: API ready", protocol.STATUS),
        ("trigger", protocol.TRIGGER),
        ("limit: off", protocol.LIMIT_OFF),
        ("stretch-motor ready", protocol.BANNER),
        ("?x", protocol.UNKNOWN_COMMAND),
        ("3A:17:95:62:00:00:00:B3", protocol.OTHER),
    ],
)
def test_parse_reply_kinds(line, kind):
    assert protocol.parse_reply(line).kind == kind


def test_parse_reply_telemetry_payloads():
    trip = protocol.parse_reply(
        "LIMIT_TRIP force_n=9.5 raw_counts=-760 zero_counts=0 calibration_n_per_count=0.0125"
    )
    assert trip.kind == protocol.LIMIT_TRIP and trip.data["force_n"] == 9.5
    old = protocol.parse_reply("LIMIT_TRIP 7.25")
    assert old.data == {"force_n": 7.25}
    armed = protocol.parse_reply(
        "LIMIT_ARMED limit_n=90.000000 release_motor_steps=200 release_dir_positive=1"
    )
    assert armed.kind == protocol.LIMIT_ARMED and armed.data["release_motor_steps"] == 200.0
    assert protocol.parse_reply("zero: -12").data == {"zero_counts": -12}


def test_overload_reason():
    assert protocol.overload_reason(None, 10) is None
    assert protocol.overload_reason(5.0, 0) is None
    assert protocol.overload_reason(float("nan"), 10) is None
    assert protocol.overload_reason(9.99, 10) is None
    assert protocol.overload_reason(-10.0, 10).startswith("force_limit_exceeded")


def test_controller_events_from_raw_bytes():
    ctrl = ArduinoController()
    lines = ctrl.feed(b"power: 3\nleds: 2 OK\nMOVE_DO")
    assert lines == ["power: 3", "leds: 2 OK"]
    assert ctrl.power_ack.is_set() and ctrl.lights_on.is_set() and not ctrl.move_done.is_set()
    ctrl.feed(
        b"NE\nLIMIT_TRIP force_n=3.5 raw_counts=1 zero_counts=0 calibration_n_per_count=0.1 limit_n=3\n"
    )
    assert ctrl.move_done.is_set() and ctrl.limit_tripped.is_set() and ctrl.last_trip_force == 3.5


def test_listeners_are_called_and_failures_are_contained():
    ctrl = ArduinoController()
    seen = []
    ctrl.add_listener(lambda line: 1 / 0)
    ctrl.add_listener(seen.append)
    ctrl.feed(b"status: API ready\n")
    assert seen == ["status: API ready"]


def test_send_requires_connection():
    with pytest.raises(ControllerError):
        ArduinoController().send("i")


def connected(**port_kwargs) -> tuple[ArduinoController, FakePort]:
    ctrl = ArduinoController(lambda *a, **k: FakePort(*a, **k, **port_kwargs))
    ctrl.connect("TEST", settle_s=0.0)

    return ctrl, ctrl.port


def test_request_tension_round_trip_and_stale_reply_is_discarded():
    ctrl, port = connected()
    try:
        ctrl.feed(
            b"TENSION force_n=99.000000 raw_counts=1 zero_counts=0 calibration_n_per_count=0.0125\n"
        )  # stale
        port.position = 2000  # 1.0 N
        reply = ctrl.request_tension(timeout=2.0)
        assert protocol.parse_force_telemetry(reply)["force_n"] == pytest.approx(1.0)
    finally:
        ctrl.disconnect()


def test_request_tension_times_out_without_reply():
    ctrl, _ = connected(answer_tension=False)
    try:
        assert ctrl.request_tension(timeout=0.2) is None
    finally:
        ctrl.disconnect()


def test_lights_context_confirms_and_always_switches_off():
    ctrl, port = connected()
    try:
        with ctrl.lights(0b101, 90):
            assert port.led_mask == 0b101 and port.power == 90
        assert port.led_mask == 0 and port.power == 0
        with pytest.raises(RuntimeError), ctrl.lights(1, 50):
            raise RuntimeError("camera failed")
        assert port.led_mask == 0 and port.power == 0
    finally:
        ctrl.disconnect()


def test_lights_context_raises_when_leds_are_never_confirmed():
    ctrl, port = connected(confirm_leds=False)
    try:
        with (
            pytest.raises(ControllerError, match="never confirmed"),
            ctrl.lights(3, 50),
        ):
            pass
        assert port.led_mask == 0 and port.power == 0
    finally:
        ctrl.disconnect()
