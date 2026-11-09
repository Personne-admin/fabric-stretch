from __future__ import annotations

import json

import numpy as np
import pytest

from fabsim import photometric
from fabsim.rig import dispatcher as dispatcher_module
from fabsim.rig import mechanics, session
from fabsim.rig.dispatcher import AutomaticDispatcher
from fabsim.rig.hardware.arduino import ArduinoController
from fabsim.rig.hardware.camera import SimCamera
from fabsim.rig.hardware.fake import FakePort
from fabsim.rig.settings import MeasurementSettings

LED = {
    "pol_a_bits": [1, 3, 5],
    "pol_b_bits": [2, 4, 6],
    "diffuse_mask": 1,
    "light_elevation_deg": 45.0,
    "light_azimuths_deg": [91.27, 334.26, 211.16],
}


class Rig:
    def __init__(self, tmp_path, **port_kwargs):
        self.ctrl = ArduinoController(lambda *a, **k: FakePort(*a, **k, **port_kwargs))
        self.ctrl.simulated = True
        self.ctrl.connect("SIM", settle_s=0.0)
        self.port = self.ctrl.port
        self.cam = SimCamera()
        self.cam.open()
        self.run_dir = tmp_path / "sample" / "warp_loading"
        self.logs: list[str] = []
        self.done: list[tuple] = []

    def dispatcher(self, settings, camera=None, **kw):
        plan = session.build_capture_plan(settings, LED)

        return AutomaticDispatcher(
            self.ctrl, camera or self.cam, self.run_dir, settings, plan, LED,
            on_log=self.logs.append, on_done=lambda *a: self.done.append(a), **kw,
        )  # fmt: skip

    def read(self, name):
        return json.loads((self.run_dir / name).read_text())


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.ctrl.disconnect()


def sequential(**kw):
    return MeasurementSettings(
        count=3, step_size=100, capture_mode="sequential", settle_ms=0, **kw
    )


def test_complete_run_writes_consistent_files(rig):
    settings = sequential(include_diffuse=True)
    d = rig.dispatcher(settings)
    d.prepare()
    d.run()

    assert rig.done == [("complete", None)]
    meta, frames, forces = (
        rig.read("session.json"),
        rig.read("frames.json"),
        rig.read("force.json"),
    )
    assert (
        meta["status"] == "complete"
        and meta["abort_reason"] is None
        and meta["frames_written"] == 3
    )
    assert (
        meta["simulated"] is True and meta["schema_version"] == session.SCHEMA_VERSION
    )
    assert meta["led_config"] == LED and [c["role"] for c in meta["capture_plan"]] == [
        "diffuse", "ps_1", "ps_2", "ps_3"
    ]  # fmt: skip
    assert len(frames["frames"]) == len(forces["forces"]) == 3
    for i, frame in enumerate(frames["frames"]):
        assert frame["frame_id"] == i and frame["step_index"] == i
        for role in ("diffuse", "ps_1", "ps_2", "ps_3"):
            assert (rig.run_dir / "images" / frame[f"image_{role}"]).is_file()

    # the physical travel: the firmware runs one command step more than commanded per move
    per_step_mm = mechanics.DEFAULT_MOTOR_MECHANICS.linear_um_per_motor_step / 1000
    assert [f["displacement_mm"] for f in frames["frames"]] == pytest.approx(
        [0, 101 * per_step_mm, 202 * per_step_mm], abs=1e-6
    )

    assert all(f["status"] == "ok" for f in forces["forces"])
    assert rig.port.led_mask == 0 and rig.port.power == 0

    # the pipeline can read the geometry straight from the snapshot
    assert photometric.load_light_config(session=meta).azimuths_deg == (
        91.27,
        334.26,
        211.16,
    )


def test_exposure_above_the_warning_threshold_does_not_break_the_run(rig):
    d = rig.dispatcher(sequential(cam_exposure_ms=25.0))
    d.prepare()
    d.run()
    assert rig.done[0][0] == "complete"
    assert rig.read("session.json")["camera"]["exposure_ms_actual"] == 25.0


def test_a_clamped_exposure_refuses_the_run(rig):
    class ClampingCamera(SimCamera):
        def set_exposure(self, ms):
            return super().set_exposure(min(ms, 12.0))

    cam = ClampingCamera()
    cam.open()
    with pytest.raises(RuntimeError, match="12 ms instead of the requested 15 ms"):
        rig.dispatcher(sequential(), camera=cam).prepare()
    assert not rig.run_dir.exists()


def test_exposure_within_tolerance_is_corrected_and_stamped_with_the_read_back_value(
    rig,
):
    from fabsim.color_calibration import CalibrationModel, ColorCalibration

    model = CalibrationModel(
        1,
        "linear",
        np.eye(3).tolist(),
        "linear",
        255.0,
        "D65",
        "sRGB D65",
        15.0,
        "RGB",
        "chart.png",
        "t",
    )

    class DriftingCamera(SimCamera):
        def set_exposure(self, ms):
            return super().set_exposure(ms * 1.01)

    cam = DriftingCamera()
    cam.open()
    d = rig.dispatcher(
        sequential(), camera=cam, color_bundle=ColorCalibration(1, model, model)
    )
    d.prepare()
    d.run()
    meta = rig.read("session.json")
    assert rig.done == [("complete", None)]
    assert meta["color_calibration"]["images_corrected"] is True
    assert meta["color_calibration"]["exposure_ms"] == pytest.approx(15.15)
    assert meta["camera"]["exposure_ms_requested"] == 15.0
    assert meta["camera"]["exposure_ms_actual"] == pytest.approx(15.15)


def test_failure_to_finalise_session_json_is_reported(rig, monkeypatch):
    d = rig.dispatcher(sequential())
    d.prepare()

    def broken(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(d.writer, "finish", broken)
    d.run()
    assert rig.done == [("error", "error: could not finalise session.json (disk full)")]
    assert any("could not finalise session.json" in line for line in rig.logs)


def test_relaxation_hold_adds_a_second_frame_for_the_same_step(rig):
    d = rig.dispatcher(
        sequential(relax_enabled=True, relax_indices=[1], relax_duration_ms=100)
    )

    d.prepare()
    d.run()
    frames = rig.read("frames.json")["frames"]
    assert [(f["step_index"], f["is_relaxation_hold"]) for f in frames] == [
        (0, False),
        (1, False),
        (1, True),
        (2, False),
    ]

    assert rig.read("force.json")["forces"][2]["label"] == "1_R"


def test_failed_capture_records_no_file_names_and_aborts(rig):
    class FlakyCamera(SimCamera):
        def capture_frame(self):
            self.calls = getattr(self, "calls", 0) + 1

            return (
                None if self.calls == 5 else super().capture_frame()
            )  # 2nd shot of the 2nd frame

    cam = FlakyCamera()
    cam.open()
    d = rig.dispatcher(sequential(), camera=cam)
    d.prepare()
    d.run()

    meta, frames = rig.read("session.json"), rig.read("frames.json")["frames"]
    assert rig.done[0][0] == "error" and meta["abort_reason"].startswith(
        "error: camera returned no frame"
    )

    assert len(frames) == 1  # the incomplete second frame is not recorded
    on_disk = {p.name for p in (rig.run_dir / "images").iterdir()}
    assert all(
        name in on_disk
        for f in frames
        for key, name in f.items()
        if key.startswith("image_")
    )
    assert rig.port.led_mask == 0 and rig.port.power == 0


def test_exception_sets_abort_reason_and_cleans_up(rig):
    class BrokenCamera(SimCamera):
        def capture_frame(self):
            raise RuntimeError("sensor on fire")

    cam = BrokenCamera()
    cam.open()
    d = rig.dispatcher(sequential(max_force_n=50.0), camera=cam)
    d.prepare()
    d.run()
    meta = rig.read("session.json")
    assert meta["status"] == "error" and meta["abort_reason"] == "error: sensor on fire"
    assert rig.done == [("error", "error: sensor on fire")]
    assert (
        rig.port.led_mask == 0 and rig.port.power == 0 and rig.port.limit_n == 0
    )  # force limit disarmed


def test_unconfirmed_leds_abort_the_run_without_a_dark_frame(tmp_path):
    r = Rig(tmp_path, confirm_leds=False)
    try:
        d = r.dispatcher(sequential())
        d.prepare()
        d.run()
        assert r.done[0][0] == "error" and "never confirmed" in r.done[0][1]
        assert r.read("frames.json")["frames"] == []
        assert r.port.led_mask == 0 and r.port.power == 0
    finally:
        r.ctrl.disconnect()


def test_firmware_force_trip_stops_the_run(rig):
    rig.port.force_per_step = 0.1  # 10 N per increment
    d = rig.dispatcher(sequential(max_force_n=15.0))
    d.prepare()
    d.run()
    meta = rig.read("session.json")
    assert (
        rig.done[0][0] == "aborted"
        and meta["abort_reason"] == "firmware_force_limit (20.0 N)"
    )
    assert meta["frames_written"] == 2


def test_firmware_trip_without_move_done_is_reported(rig):
    # The real firmware ends a tripped move without MOVE_DONE.
    rig.port.force_per_step = 0.1
    rig.port.limit_n = 15.0
    replies = rig.port._answer("+200")
    assert (
        any(line.startswith("LIMIT_TRIP") for line in replies)
        and "MOVE_DONE" not in replies
    )

    rig.port.limit_n = 0.0
    rig.port.position = 0
    d = rig.dispatcher(sequential(max_force_n=15.0))
    d.prepare()
    d.run()
    assert rig.read("session.json")["abort_reason"] == "firmware_force_limit (20.0 N)"
    assert rig.port.limit_n == 0  # disarmed afterwards


def test_a_move_that_never_completes_aborts_with_move_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher_module, "MOVE_TIMEOUT_SLACK_S", 0.0)
    r = Rig(tmp_path, answer_moves=False)
    try:
        d = r.dispatcher(sequential())
        d.prepare()
        d.run()
        meta = r.read("session.json")
        assert r.done[0][0] == "aborted" and meta["abort_reason"].startswith(
            "move_timeout"
        )
        assert meta["frames_written"] == 1 and r.port.led_mask == 0
    finally:
        r.ctrl.disconnect()


def test_unarmed_force_limit_aborts_before_the_first_move(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatcher_module, "LIMIT_ARM_TIMEOUT_S", 0.05)
    r = Rig(tmp_path, answer_limit=False)
    try:
        d = r.dispatcher(sequential(max_force_n=15.0))
        d.prepare()
        d.run()
        assert r.done[0][0] == "aborted" and r.read("session.json")[
            "abort_reason"
        ].startswith("force_limit_not_armed")
        assert r.read("session.json")["frames_written"] == 0
        assert not any(cmd.startswith(("+", "-")) for cmd in r.port.log)
    finally:
        r.ctrl.disconnect()


def test_a_trip_does_not_abort_the_next_run_on_the_same_connection(rig):
    rig.port.force_per_step = 0.1
    first = rig.dispatcher(sequential(max_force_n=15.0))
    first.prepare()
    first.run()
    assert rig.read("session.json")["abort_reason"].startswith("firmware_force_limit")

    rig.port.position = 0
    rig.run_dir = rig.run_dir.with_name("warp_release")
    second = rig.dispatcher(sequential(motion_mode="release", max_force_n=0.0))
    second.prepare()
    second.run()
    assert rig.done[-1] == ("complete", None)
    assert rig.read("session.json")["frames_written"] == 3


def test_software_force_limit_stops_the_run(tmp_path):
    r = Rig(tmp_path, force_per_step=0.1, firmware_trips=False)
    try:
        d = r.dispatcher(sequential(max_force_n=15.0))
        d.prepare()
        d.run()
        meta = r.read("session.json")
        assert r.done[0][0] == "aborted" and meta["abort_reason"].startswith(
            "force_limit_exceeded"
        )
        assert meta["frames_written"] == 3  # the frame that exceeded the limit is kept
    finally:
        r.ctrl.disconnect()


def test_stretch_limit_stops_before_the_move(rig):
    d = rig.dispatcher(sequential(max_stretch_mm=0.02))
    d.prepare()
    d.run()
    assert rig.read("session.json")["abort_reason"].startswith("stretch_limit_exceeded")
    assert rig.read("session.json")["frames_written"] == 2


def test_cancel_before_start_is_recorded(rig):
    d = rig.dispatcher(sequential())
    d.prepare()
    d.cancel("user_stop")
    d.run()
    assert rig.read("session.json")["abort_reason"] == "user_stop"
    assert rig.done[0][0] == "aborted"


def test_threaded_run_and_progress_callbacks(rig):
    progress = []
    d = rig.dispatcher(
        sequential(), on_progress=lambda cur, total: progress.append((cur, total))
    )
    d.start()
    d.join(timeout=20)
    assert not d.running and progress == [(1, 3), (2, 3), (3, 3)]
    assert rig.done == [("complete", None)]


def test_existing_run_is_never_overwritten(rig):
    first = rig.dispatcher(sequential())
    first.prepare()
    first.run()
    again = rig.dispatcher(sequential())
    with pytest.raises(FileExistsError):
        again.prepare()
    assert rig.read("session.json")["status"] == "complete"


def test_run_requires_an_open_camera_and_a_connection(rig):
    closed = SimCamera()
    with pytest.raises(RuntimeError, match="camera"):
        rig.dispatcher(sequential(), camera=closed).prepare()
    rig.ctrl.disconnect()
    with pytest.raises(RuntimeError, match="controller"):
        rig.dispatcher(sequential()).prepare()
    assert not rig.run_dir.exists()


def test_invalid_settings_are_rejected_before_anything_is_written(rig):
    with pytest.raises(ValueError, match="Captures"):
        rig.dispatcher(MeasurementSettings(count=0)).prepare()
    assert not rig.run_dir.exists()


def test_dark_frame_statistic_is_kept_per_role(rig):
    d = rig.dispatcher(sequential())
    bright = np.full((8, 8, 3), 200, np.uint8)
    dark = np.full((8, 8, 3), 20, np.uint8)
    from PIL import Image

    d._check_brightness(Image.fromarray(bright), "ps_1", "a.png")
    d._check_brightness(
        Image.fromarray(dark), "ps_b_1", "b.png"
    )  # a different role: no warning
    assert not any("missed illumination" in line for line in rig.logs)
    d._check_brightness(Image.fromarray(dark), "ps_1", "c.png")
    assert any("missed illumination" in line for line in rig.logs)
    assert set(d._role_brightness) == {"ps_1", "ps_b_1"}


def test_colour_correction_stamp_and_raw_copies(rig, tmp_path):
    from fabsim.color_calibration import CalibrationModel, ColorCalibration

    def model(exposure):
        return CalibrationModel(
            version=1,
            fit_type="linear",
            coefficients=np.eye(3).tolist(),
            input_encoding="linear",
            input_scale=255.0,
            reference_white="D65",
            output_space="sRGB D65",
            exposure_ms=exposure,
            channel_order="RGB",
            source_file="C:/some/where/chart.png; chart2.png",
            fitted_at="2026-01-01T00:00:00Z",
        )

    matching = ColorCalibration(1, model(15.0), model(15.0))
    d = rig.dispatcher(
        sequential(include_diffuse=True), color_bundle=matching, keep_raw=True
    )
    d.prepare()
    d.run()
    stamp = rig.read("session.json")["color_calibration"]
    assert stamp["images_corrected"] is True and stamp["applied"] is True
    assert stamp["calibration_source"] == "chart.png; chart2.png"  # no directories
    assert (rig.run_dir / "color_calibration.json").is_file()
    assert len(list((rig.run_dir / "images_raw").iterdir())) == 12

    assert stamp["exposure_ms"] == 15.0
    assert "chart.png" in rig.read("color_calibration.json")["diffuse"]["source_file"]
    assert (
        "/" not in rig.read("color_calibration.json")["diffuse"]["source_file"]
    )  # no directories

    # A bundle for another exposure is not applied, and the stamp says so explicitly.
    other = Rig(tmp_path / "other")
    try:
        mismatched = ColorCalibration(1, model(25.0), model(25.0))
        d2 = other.dispatcher(sequential(), color_bundle=mismatched)
        d2.prepare()
        d2.run()
        stamp = other.read("session.json")["color_calibration"]
        assert stamp["images_corrected"] is False and "uncorrected" in stamp["reason"]
        assert not (other.run_dir / "images_raw").exists()
    finally:
        other.ctrl.disconnect()
