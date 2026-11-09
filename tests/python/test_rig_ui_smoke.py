from __future__ import annotations

import json
import sys
import time
import types
from unittest.mock import MagicMock

import pytest

from fabsim.rig import config


class TclError(Exception):
    pass


class FakeVar:
    default = ""

    def __init__(self, master=None, value=None, name=None):
        self._value = self.default if value is None else value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class FakeWidget:
    def __init__(self, *args, **kwargs):
        self._options = dict(kwargs)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        method = MagicMock(name=name)
        object.__setattr__(self, name, method)

        return method

    def __setitem__(self, key, value):
        self._options[key] = value

    def __getitem__(self, key):
        return self._options[key]

    def after(self, *args, **kwargs):
        return "after-id"

    def winfo_children(self):
        return []

    def curselection(self):
        return ()


class FakeModule(types.ModuleType):
    classes: dict

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        if name == "TclError":
            return TclError

        if name.endswith("Var"):
            default = {"IntVar": 0, "DoubleVar": 0.0, "BooleanVar": False}.get(name, "")
            cls = type(name, (FakeVar,), {"default": default})
        elif name in {
            "showinfo",
            "showwarning",
            "showerror",
            "askyesno",
            "askstring",
            "askdirectory",
        }:
            cls = MagicMock(
                name=name, return_value=True if name == "askyesno" else None
            )
        else:
            cls = type(name, (FakeWidget,), {})

        setattr(self, name, cls)

        return cls


_TK_DEPENDENT = ("fabsim.rig.ui", "fabsim.rig.stereo", "PIL.ImageTk")


def _is_tk_dependent(name: str) -> bool:
    return any(name == n or name.startswith(n + ".") for n in _TK_DEPENDENT)


@pytest.fixture
def fake_tk():
    import fabsim.rig

    saved = {
        n: m
        for n, m in sys.modules.items()
        if n.startswith("tkinter") or _is_tk_dependent(n)
    }
    saved_attrs = {
        a: getattr(fabsim.rig, a) for a in ("ui", "stereo") if hasattr(fabsim.rig, a)
    }
    for name in saved:
        del sys.modules[name]

    for attr in saved_attrs:
        delattr(fabsim.rig, attr)

    modules = {"tkinter": FakeModule("tkinter")}
    for sub in ("ttk", "filedialog", "messagebox", "scrolledtext", "simpledialog"):
        modules[f"tkinter.{sub}"] = FakeModule(f"tkinter.{sub}")
        setattr(modules["tkinter"], sub, modules[f"tkinter.{sub}"])

    sys.modules.update(modules)

    try:
        yield modules
    finally:
        for name in [
            n for n in sys.modules if n.startswith("tkinter") or _is_tk_dependent(n)
        ]:
            del sys.modules[name]
        for attr in ("ui", "stereo"):
            if hasattr(fabsim.rig, attr):
                delattr(fabsim.rig, attr)

        sys.modules.update(saved)
        for attr, module in saved_attrs.items():
            setattr(fabsim.rig, attr, module)


def pump(app, until, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        while not app._events.empty():
            app._events.get_nowait()()

        if until():
            return
        time.sleep(0.01)

    raise AssertionError("timed out waiting for the UI event")


@pytest.fixture
def app(fake_tk, monkeypatch, tmp_path):
    monkeypatch.setenv("FABSIM_RIG_CONFIG_DIR", str(tmp_path / "config"))
    from fabsim.rig.ui import main_window, widgets

    monkeypatch.setattr(
        widgets.ImageCanvas,
        "show_image",
        lambda self, image: setattr(self, "_image", image),
    )
    window = main_window.App()
    window.canvas.clear = MagicMock()
    yield window
    window.ctrl.disconnect()


def connect_simulated(app):
    app.port_var.set("SIMULATED")
    app._toggle_connect()
    pump(app, lambda: app.ctrl.connected)
    assert app.ctrl.simulated


def test_window_builds_and_uses_packaged_defaults_without_writing(app, tmp_path):
    assert app._presets and app.led_config["pol_a_bits"]
    assert not (tmp_path / "config").exists()
    assert app.ps_tab is not None
    assert "SIMULATED" in app.port_combo["values"]


def test_simulated_measurement_end_to_end(app, tmp_path, fake_tk):
    connect_simulated(app)
    app.sim_camera_var.set(True)
    app._toggle_camera()
    assert app.cam.is_open and app.cam.simulated

    app.project_dir = tmp_path / "project"
    app.count_var.set(3)
    app.stepsize_var.set(50)
    app.settle_var.set(0)
    app.sample_id_var.set("cloth")
    app.mode_var.set("sequential")
    app.camexp_var.set(25.0)
    app._run_measurement()
    assert app.runner is not None
    pump(
        app,
        lambda: (
            app.lbl_progress.config.call_args
            and app.lbl_progress.config.call_args.kwargs.get("text") == "Complete"
        ),
    )

    run = app.project_dir / "cloth" / "warp_loading"
    meta = json.loads((run / "session.json").read_text())
    assert (
        meta["status"] == "complete"
        and meta["simulated"] is True
        and meta["frames_written"] == 3
    )
    assert len(list((run / "images").glob("*_ps_*.png"))) == 9
    assert any(
        "camera exposure is 20 ms or higher" in c.args[1]
        for c in app.log_text.insert.call_args_list
    )

    app.lbl_progress.config.reset_mock()
    app._run_measurement()
    pump(
        app,
        lambda: (
            app.lbl_progress.config.call_args
            and app.lbl_progress.config.call_args.kwargs.get("text") == "Complete"
        ),
    )
    assert (app.project_dir / "cloth" / "warp_loading_2" / "session.json").is_file()
    assert json.loads((run / "session.json").read_text()) == meta


def test_run_refuses_without_camera_or_project(app, fake_tk, tmp_path):
    connect_simulated(app)
    app.project_dir = tmp_path
    app._run_measurement()  # camera closed
    assert app.runner is None
    assert fake_tk["tkinter.messagebox"].showwarning.called


def test_preset_save_writes_the_user_file(app, fake_tk, tmp_path):
    fake_tk["tkinter.simpledialog"].askstring.return_value = "my preset"
    app.count_var.set(7)
    app._save_preset()
    stored = json.loads((tmp_path / "config" / config.PRESETS_FILE).read_text())
    assert stored[-1]["name"] == "my preset" and stored[-1]["count"] == 7


def test_led_mapping_save_keeps_the_light_geometry(app, tmp_path):
    azimuths = list(app.led_config["light_azimuths_deg"])
    app.update_led_mapping(app.led_config["pol_a_bits"], [2, 4, 7], 0b11)
    saved = json.loads((tmp_path / "config" / config.LED_CONFIG_FILE).read_text())
    assert saved["light_azimuths_deg"] == azimuths and saved["pol_b_bits"] == [2, 4, 7]
    assert saved["light_elevation_deg"] == app.led_config["light_elevation_deg"]

    app.save_light_geometry(40.0, [1.0, 2.0, 3.0])
    saved = json.loads((tmp_path / "config" / config.LED_CONFIG_FILE).read_text())
    assert saved["light_azimuths_deg"] == [1.0, 2.0, 3.0] and saved["pol_b_bits"] == [
        2,
        4,
        7,
    ]


def test_changing_set_a_clears_the_azimuths_and_warns(app, tmp_path):
    app.update_led_mapping([0, 2, 4], [], 0)
    saved = json.loads((tmp_path / "config" / config.LED_CONFIG_FILE).read_text())
    assert "light_azimuths_deg" not in saved
    assert any(
        "azimuths were cleared" in c.args[1] for c in app.log_text.insert.call_args_list
    )


def test_led_identifier_role_mapping(fake_tk):
    from fabsim.rig.ui import led_identifier

    roles = {
        0: "pol_b_2",
        1: "pol_a_0",
        2: "pol_b_0",
        3: "pol_a_1",
        4: "pol_b_1",
        5: "pol_a_2",
        6: "diffuse",
        7: "diffuse",
    }
    assert led_identifier.roles_to_mapping(roles) == ([1, 3, 5], [2, 4, 0], 0b11000000)
    cfg = {"pol_a_bits": [1, 3, 5], "pol_b_bits": [2, 4, 0], "diffuse_mask": 0b11000000}
    assert [led_identifier.bit_role(cfg, b) for b in range(8)] == [
        "pol_b_2", "pol_a_0", "pol_b_0", "pol_a_1", "pol_b_1", "pol_a_2", "diffuse", "diffuse"
    ]  # fmt: skip
    assert led_identifier.roles_to_mapping({0: "unused", 1: "diffuse"}) == (
        [],
        [],
        0b10,
    )


def test_led_identifier_rejects_incomplete_sets(fake_tk):
    from fabsim.rig.ui import led_identifier

    # A2 and A3 without A1 would be captured as ps_1/ps_2 and mislabel the LEDs.
    with pytest.raises(ValueError, match="Set A.*A1"):
        led_identifier.roles_to_mapping(
            {1: "unused", 2: "pol_a_1", 3: "pol_a_2", 4: "diffuse"}
        )

    with pytest.raises(ValueError, match="Set B"):
        led_identifier.roles_to_mapping(
            {0: "pol_a_0", 1: "pol_a_1", 2: "pol_a_2", 3: "pol_b_0"}
        )


def test_loading_a_preset_keeps_the_capture_setup_it_does_not_define(app, fake_tk):
    app.mode_var.set("sequential")
    app.polarization_var.set(True)
    app.include_diffuse_var.set(True)
    app._preset_listbox.curselection = lambda: (0,)
    app._presets[0] = {"name": "legacy", "count": 5}  # no capture setup keys
    app._load_preset()
    assert app.count_var.get() == 5
    assert (
        app.mode_var.get() == "sequential"
        and app.polarization_var.get()
        and app.include_diffuse_var.get()
    )

    app._presets[0] = {
        "name": "full",
        "count": 6,
        "capture_mode": "rgb",
        "polarization": False,
    }
    app._load_preset()
    assert app.mode_var.get() == "rgb" and not app.polarization_var.get()


def test_controls_that_disturb_a_run_are_locked_while_it_runs(app, fake_tk):
    app._set_run_locked(True)
    for widget in (
        app.btn_cam,
        app.btn_connect,
        app._speed_btn,
        app._raw_entry,
        app._raw_send,
    ):
        widget.state.assert_called_with(["disabled"])

    assert len(app._dialog_buttons) == 3
    for widget in app._dialog_buttons:
        widget.state.assert_called_with(["disabled"])

    app._set_run_locked(False)
    app._speed_btn.state.assert_called_with(["!disabled"])


def test_speed_cannot_change_during_a_run(app, fake_tk):
    class Running:
        running = True

    app.runner = Running()
    before = app.speed_var.get()
    app._toggle_speed()
    assert app.speed_var.get() == before


def test_run_refuses_uncalibrated_light_directions(app, fake_tk, tmp_path):
    connect_simulated(app)
    app.sim_camera_var.set(True)
    app._toggle_camera()
    app.project_dir = tmp_path
    app.mode_var.set("sequential")
    app.led_config = {
        k: v for k, v in app.led_config.items() if k != "light_azimuths_deg"
    }

    app._run_measurement()
    assert app.runner is None
    assert (
        "Calibrate Light Angles"
        in fake_tk["tkinter.messagebox"].showerror.call_args.args[1]
    )


def test_light_calibration_geometry(fake_tk):
    from fabsim.rig.ui.light_calibration import shadow_to_light_azimuth

    assert shadow_to_light_azimuth((10, 10), (20, 10)) == pytest.approx(180.0)
    assert shadow_to_light_azimuth((10, 10), (10, 20)) == pytest.approx(
        90.0
    )  # shadow down -> light up
    assert shadow_to_light_azimuth((10, 10), (0, 10)) == pytest.approx(0.0)


def test_ps_tab_loads_a_run_and_previews(app, fake_tk, tmp_path):
    from test_rig_ps_process import write_run

    run, _, _ = write_run(tmp_path)
    tab = app.ps_tab
    tab._load_run_folder(run)
    assert tab.run is not None and "session.json:led_config" in tab.status_var.get()
    tab._preview_current()
    pump(
        app,
        lambda: tab.canvas_alb.show_image is not None
        and tab.canvas_alb._image is not None,
    )

    tab._process_all()
    pump(
        app,
        lambda: (run / "ps_output" / "ps_manifest.json").exists()
        and tab._cancel_event is None,
    )

    manifest = json.loads((run / "ps_output" / "ps_manifest.json").read_text())
    assert (
        manifest["solver"] == "fabsim.photometric.solve"
        and len(manifest["frames"]) == 2
    )

    assert (run / "ps_output" / "img_0001_normals.png").is_file()


def test_ps_tab_reports_a_run_without_azimuths(app, fake_tk, tmp_path):
    from test_rig_ps_process import LED, write_run

    run, _, _ = write_run(
        tmp_path, led={k: v for k, v in LED.items() if k != "light_azimuths_deg"}
    )

    app.ps_tab._load_run_folder(run)
    assert app.ps_tab.run is None
    assert fake_tk["tkinter.messagebox"].showerror.called


def test_entry_points_import_without_tkinter():
    import subprocess

    code = "import sys, fabsim.rig.app, fabsim.rig.__main__; sys.exit(1 if 'tkinter' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
