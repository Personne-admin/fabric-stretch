from __future__ import annotations

import json
import logging
import sys

import pytest

from fabsim import photometric
from fabsim.rig import config
from fabsim.rig.settings import MeasurementSettings


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "FABSIM_RIG_CONFIG_DIR",
        "FABSIM_RIG_LOG_DIR",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "APPDATA",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_dir_override_and_xdg(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config.config_dir() == tmp_path / "xdg" / "fabsim-rig"
    monkeypatch.setenv("FABSIM_RIG_CONFIG_DIR", str(tmp_path / "custom"))
    assert config.config_dir() == (tmp_path / "custom").resolve()


def test_default_dirs_are_in_the_home_directory(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(config, "_home", lambda: tmp_path)
    assert config.config_dir() == tmp_path / ".config" / "fabsim-rig"
    assert config.log_dir() == tmp_path / ".local" / "state" / "fabsim-rig"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert config.config_dir() == tmp_path / "Roaming" / "fabsim-rig"


def test_paths_do_not_depend_on_the_working_directory(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_home", lambda: tmp_path / "home")
    before = (config.config_dir(), config.log_dir())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert (config.config_dir(), config.log_dir()) == before
    assert all(path.is_absolute() for path in before)
    config.load_led_config(tmp_path / "nonexistent")  # packaged default, writes nothing
    assert list(elsewhere.iterdir()) == []


def test_missing_user_files_fall_back_to_packaged_defaults_without_writing(tmp_path):
    directory = tmp_path / "cfg"
    led = config.load_led_config(directory)
    presets = config.load_presets(directory)
    assert led["pol_a_bits"] and len(led["light_azimuths_deg"]) == 3
    assert presets and all("count" in p for p in presets)
    assert not directory.exists()


def test_packaged_led_config_is_what_photometric_uses():
    # There is exactly one default: the file in fabsim/rig/defaults.
    led = json.loads(config.packaged_default("led_config.json").read_text())
    cfg = photometric.load_light_config()
    assert list(cfg.azimuths_deg) == led["light_azimuths_deg"]
    assert cfg.elevation_deg == led["light_elevation_deg"]


def test_user_file_wins_and_corrupt_user_file_is_an_error(tmp_path):
    config.save_led_config({"pol_a_bits": [0, 1, 2], "light_azimuths_deg": [1, 2, 3]}, tmp_path)
    assert config.load_led_config(tmp_path)["pol_a_bits"] == [0, 1, 2]
    (tmp_path / config.LED_CONFIG_FILE).write_text("{not json")
    with pytest.raises(config.ConfigError):
        config.load_led_config(tmp_path)


def test_seed_defaults_is_explicit_and_does_not_overwrite(tmp_path):
    written = config.seed_defaults(tmp_path)
    assert {p.name for p in written} == set(config.DEFAULT_FILES)
    (tmp_path / config.PRESETS_FILE).write_text("[]")
    assert config.seed_defaults(tmp_path) == []
    assert (tmp_path / config.PRESETS_FILE).read_text() == "[]"
    assert len(config.seed_defaults(tmp_path, overwrite=True)) == len(config.DEFAULT_FILES)


def test_merge_led_config_keeps_geometry_when_set_a_is_unchanged():
    current = {
        "pol_a_bits": [1, 3, 5],
        "pol_b_bits": [2, 4, 6],
        "diffuse_mask": 1,
        "light_elevation_deg": 41.5,
        "light_azimuths_deg": [91.27, 334.26, 211.16],
    }
    merged = config.merge_led_config(current, [1, 3, 5], [2, 4, 7], 0b11)
    assert merged["light_azimuths_deg"] == [91.27, 334.26, 211.16]
    assert merged["light_elevation_deg"] == 41.5
    assert merged["pol_b_bits"] == [2, 4, 7] and merged["diffuse_mask"] == 3
    assert current["pol_b_bits"] == [2, 4, 6]  # the input is not modified


def test_merge_led_config_drops_azimuths_only_if_set_a_changed():
    current = {
        "pol_a_bits": [1, 3, 5],
        "light_elevation_deg": 45.0,
        "light_azimuths_deg": [10, 20, 30],
    }
    merged = config.merge_led_config(current, [3, 1, 5], [], 0)
    assert "light_azimuths_deg" not in merged
    assert merged["light_elevation_deg"] == 45.0
    with pytest.raises(ValueError):  # and the PS tools refuse to guess
        photometric.load_light_config(session={"led_config": merged})


def test_state_round_trip_and_color_calibration_lookup(tmp_path):
    assert config.default_color_calibration_path(tmp_path) is None
    (tmp_path / config.COLOR_CALIBRATION_FILE).write_text("{}")
    assert config.default_color_calibration_path(tmp_path) == tmp_path / config.COLOR_CALIBRATION_FILE
    elsewhere = tmp_path / "other.json"
    elsewhere.write_text("{}")
    config.update_state(tmp_path, color_calibration=str(elsewhere))
    assert config.default_color_calibration_path(tmp_path) == elsewhere
    config.update_state(tmp_path, project_dir="/data")
    assert config.load_state(tmp_path) == {
        "color_calibration": str(elsewhere),
        "project_dir": "/data",
    }


def test_presets_round_trip_into_settings(tmp_path):
    for preset in config.load_presets(tmp_path / "empty"):  # packaged defaults
        settings = MeasurementSettings.from_preset(preset)
        settings.validate()
        assert settings.step_size >= 0
    settings = MeasurementSettings(count=5, step_size=120, motion_mode="release", relax_indices=[3, 1])
    path = config.save_presets([settings.to_preset("mine")], tmp_path)
    assert path == tmp_path / "presets.json"
    [stored] = config.load_presets(tmp_path)
    assert stored["name"] == "mine" and stored["relax_indices"] == [1, 3]
    assert MeasurementSettings.from_preset(stored) == MeasurementSettings(
        count=5, step_size=120, motion_mode="release", relax_indices=[1, 3]
    )


def test_legacy_signed_step_size_preset():
    settings = MeasurementSettings.from_preset({"count": 4, "step_size": -250})
    assert (settings.step_size, settings.motion_mode) == (250, "release")
    settings = MeasurementSettings.from_preset(
        {"commanded_motor_steps_per_increment": 90, "motion_mode": "loading"}
    )
    assert (settings.step_size, settings.motion_mode) == (90, "loading")


def test_settings_validation_lists_all_problems():
    bad = MeasurementSettings(count=0, led_power=300, cam_exposure_ms=0, sample_id="a/b")
    with pytest.raises(ValueError) as info:
        bad.validate()
    message = str(info.value)
    assert (
        "Captures" in message and "LED power" in message and "exposure" in message and "Sample ID" in message
    )


def test_setup_logging_writes_a_rotating_file(tmp_path):
    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        path = config.setup_logging(tmp_path / "logs")
        logging.getLogger("fabsim.test").info("hello rig")
        for handler in root.handlers:
            handler.flush()
        assert path == tmp_path / "logs" / config.LOG_FILE
        assert "hello rig" in path.read_text()
        config.setup_logging(tmp_path / "logs")  # calling twice does not duplicate handlers
        assert sum(getattr(h, "_fabsim_rig", False) for h in root.handlers) == 2
    finally:
        for handler in list(root.handlers):
            if getattr(handler, "_fabsim_rig", False):
                handler.close()
                root.removeHandler(handler)
        for handler in saved:
            if handler not in root.handlers:
                root.addHandler(handler)


def test_preset_without_capture_setup_keeps_the_current_one():
    from fabsim.rig.settings import MeasurementSettings

    current = MeasurementSettings(capture_mode="sequential", polarization=True, release_steps=7)
    legacy = MeasurementSettings.from_preset({"name": "old", "count": 4}, current)
    assert (legacy.count, legacy.capture_mode, legacy.polarization, legacy.release_steps) == (
        4, "sequential", True, 7,
    )  # fmt: skip
    explicit = MeasurementSettings.from_preset({"capture_mode": "rgb", "polarization": False}, current)
    assert explicit.capture_mode == "rgb" and not explicit.polarization and explicit.release_steps == 7
    assert MeasurementSettings.from_preset({"count": 4}).capture_mode == "rgb"  # no current: defaults
