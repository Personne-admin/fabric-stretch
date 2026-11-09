from __future__ import annotations

import json

import pytest

from fabsim import photometric
from fabsim.rig import config, session
from fabsim.rig.colorcorrect import corrected_models
from fabsim.rig.settings import MeasurementSettings

LED = {
    "pol_a_bits": [1, 3, 5],
    "pol_b_bits": [2, 4, 6],
    "diffuse_mask": 1,
    "light_elevation_deg": 45.0,
    "light_azimuths_deg": [91.27, 334.26, 211.16],
}


def test_rgb_plan_is_one_shot_with_the_typed_mask():
    plan = session.build_capture_plan(MeasurementSettings(led_mask=85), LED)
    assert plan == [("rgb", 85)]


def test_sequential_plan_orders_diffuse_then_set_a_then_set_b():
    s = MeasurementSettings(
        capture_mode="sequential", include_diffuse=True, polarization=True
    )

    assert session.build_capture_plan(s, LED) == [
        ("diffuse", 1),
        ("ps_1", 0b10),
        ("ps_2", 0b1000),
        ("ps_3", 0b100000),
        ("ps_b_1", 0b100),
        ("ps_b_2", 0b10000),
        ("ps_b_3", 0b1000000),
    ]

    plain = session.build_capture_plan(
        MeasurementSettings(capture_mode="sequential"), LED
    )

    assert [role for role, _ in plain] == ["ps_1", "ps_2", "ps_3"]


def test_plan_errors_instead_of_guessing():
    s = MeasurementSettings(capture_mode="sequential")
    with pytest.raises(session.CapturePlanError):
        session.build_capture_plan(s, {"pol_a_bits": []})

    with pytest.raises(session.CapturePlanError, match="diffuse"):
        session.build_capture_plan(
            MeasurementSettings(capture_mode="sequential", include_diffuse=True),
            {**LED, "diffuse_mask": 0},
        )


def test_sequential_plan_needs_complete_sets_and_calibrated_directions():
    s = MeasurementSettings(capture_mode="sequential")
    with pytest.raises(session.CapturePlanError, match="Set A needs exactly three"):
        session.build_capture_plan(
            s, {**LED, "pol_a_bits": [2, 3]}
        )  # A2, A3 without A1

    no_azimuths = {k: v for k, v in LED.items() if k != "light_azimuths_deg"}
    with pytest.raises(session.CapturePlanError, match="Calibrate Light Angles"):
        session.build_capture_plan(s, no_azimuths)

    with pytest.raises(session.CapturePlanError, match="Set B"):
        session.build_capture_plan(
            MeasurementSettings(capture_mode="sequential", polarization=True),
            {**LED, "pol_b_bits": [2]},
        )

    # the RGB mode needs no geometry
    assert session.build_capture_plan(MeasurementSettings(), {}) == [("rgb", 85)]


def test_run_dir_layout_and_suffix_for_an_existing_run(tmp_path):
    s = MeasurementSettings(
        sample_id=" cloth ", direction="weft", motion_mode="release"
    )

    base = session.run_dir_for(tmp_path, s)
    assert base == tmp_path / "cloth" / "weft_release"
    assert session.next_free_run_dir(base) == base
    base.mkdir(parents=True)
    assert session.next_free_run_dir(base) == base  # an empty folder is not a run
    (base / "session.json").write_text("{}")
    assert session.next_free_run_dir(base) == base.with_name("weft_release_2")
    (base.with_name("weft_release_2")).mkdir()
    (base.with_name("weft_release_2") / "frames.json").write_text("{}")
    assert session.next_free_run_dir(base) == base.with_name("weft_release_3")


def make_meta(settings=None, **kw):
    settings = settings or MeasurementSettings(capture_mode="sequential")
    args = {
        "test_id": "warp_stretch_20260101_000000",
        "plan": session.build_capture_plan(settings, LED),
        "led_config": LED,
        "camera_info": {"model": "test"},
        "color_stamp": {"images_corrected": False},
        "simulated": False,
    }

    args.update(kw)

    return session.build_session_meta(settings, **args)


def test_color_bundle_copy_and_stamp_contain_no_directories(tmp_path):
    from fabsim.color_calibration import CalibrationModel, ColorCalibration, load_bundle
    from fabsim.rig.colorcorrect import ColorCorrector, basenames

    assert basenames("C:/Users/x/charts/a.png; /data/y/b.png") == "a.png; b.png"
    model = CalibrationModel(
        1, "linear", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "linear", 255.0, "D65", "sRGB D65", 15.0, "RGB",
        "C:\\Users\\x\\a.png; /home/y/b.png", "t",
    )  # fmt: skip
    corrector = ColorCorrector(ColorCalibration(1, model, model), 15.1, True, tmp_path)
    corrector.save_bundle_copy()
    stored = load_bundle(tmp_path / "color_calibration.json")
    assert (
        stored.diffuse.source_file == stored.point_light.source_file == "a.png; b.png"
    )
    assert corrector.stamp["calibration_source"] == "a.png; b.png"


def test_session_meta_snapshots_what_the_pipeline_needs():
    meta = make_meta(
        MeasurementSettings(
            capture_mode="sequential", step_size=500, motion_mode="release"
        )
    )

    assert meta["schema_version"] == session.SCHEMA_VERSION
    assert meta["led_config"] == LED
    assert meta["capture_plan"][0] == {"role": "ps_1", "mask": 2}
    assert (
        meta["step_size"] == -500 and meta["commanded_motor_steps_per_increment"] == 500
    )

    assert meta["branch"] == meta["motion_mode"] == "release"
    assert meta["led_mask"] is None  # meaningless outside RGB mode
    assert meta["microsteps"] == 1 and meta["driver_pulses_per_command_step"] == 16
    assert (
        meta["software"]["python"] and meta["started_at"][-6] in "+-"
    )  # timezone offset

    assert meta["simulated"] is False and meta["status"] == "running"
    # The pipeline resolves the light geometry from this very snapshot.
    cfg = photometric.load_light_config(session=meta)
    assert (
        cfg.azimuths_deg == (91.27, 334.26, 211.16)
        and cfg.source == "session.json:led_config"
    )


def test_session_meta_is_json_serialisable():
    json.dumps(make_meta())


def test_writer_persists_every_frame_and_finishes_the_run(tmp_path):
    run = tmp_path / "run"
    writer = session.SessionWriter(run, make_meta())
    writer.start()
    assert (run / "images").is_dir() and json.loads((run / "session.json").read_text())[
        "status"
    ] == "running"
    assert json.loads((run / "frames.json").read_text())["frames"] == []

    force = session.blank_force_record(0, "0")
    writer.add_frame(
        session.frame_record(
            0, 0, 0.0, 0, None, {"image_ps_1": "img_0000_ps_1.png"}, 50, False
        ),
        force,
    )
    frames = json.loads((run / "frames.json").read_text())
    assert [f["image_ps_1"] for f in frames["frames"]] == ["img_0000_ps_1.png"]
    assert (
        frames["frames"][0]["force_csv_row"]
        == frames["frames"][0]["force_json_index"]
        == 0
    )
    assert (
        json.loads((run / "force.json").read_text())["forces"][0]["status"] == "timeout"
    )
    assert json.loads((run / "session.json").read_text())["frames_written"] == 1

    writer.finish("user_stop", "aborted")
    meta = json.loads((run / "session.json").read_text())
    assert (meta["status"], meta["abort_reason"], meta["finished_at"] is not None) == (
        "aborted",
        "user_stop",
        True,
    )
    assert not list(run.glob("*.tmp"))


def test_writer_refuses_to_overwrite_an_existing_run(tmp_path):
    run = tmp_path / "run"
    session.SessionWriter(run, make_meta()).start()
    with pytest.raises(FileExistsError):
        session.SessionWriter(run, make_meta()).start()


def test_line_load_and_force_record():
    assert session.line_load_n_per_m(2.0, 0.2) == pytest.approx(10.0)
    assert session.line_load_n_per_m(None, 0.2) is None
    assert session.line_load_n_per_m(2.0, 0) is None


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        (None, set()),
        ({"images_corrected": True}, {"diffuse", "point_light"}),
        ({"images_corrected": False, "applied": False}, set()),
        ({"applied": True}, {"diffuse", "point_light"}),
        ({"applied": False}, set()),
        # sessions written before the explicit flag
        (
            {"applied": True, "reason": "exposure mismatch, images saved uncorrected"},
            set(),
        ),
        (
            {
                "applied": True,
                "reason": "exposure mismatch, images saved uncorrected",
                "roles": "diffuse corrected post-hoc (diffuse model); ps_* not corrected",
            },
            {"diffuse"},
        ),
    ],
)
def test_corrected_models_reads_old_and_new_stamps(stamp, expected):
    assert corrected_models(stamp) == expected


def test_default_presets_carry_no_personal_paths():
    text = config.packaged_default("presets.json").read_text()
    assert "C:" not in text and "/home" not in text
