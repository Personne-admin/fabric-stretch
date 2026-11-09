from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

from PIL import Image

from .. import config, session
from ..dispatcher import AutomaticDispatcher
from ..hardware.arduino import ArduinoController
from ..hardware.camera import SimCamera, UEyeCamera, frame_to_pil
from ..hardware.fake import SIMULATED_PORT, FakePort
from ..settings import EXPOSURE_WARN_MS, MeasurementSettings
from .color_calibration_dialog import ColorCalibrationDialog
from .led_identifier import LedIdentifierDialog
from .light_calibration import LightCalibrationDialog
from .relax_points import RelaxPointsDialog
from .widgets import ImageCanvas, ScrollableFrame, ThreadBridge

log = logging.getLogger(__name__)

TITLE = "Fabric Stretch Rig"


class App(ThreadBridge, tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITLE)
        self.geometry("1280x820")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._init_bridge()

        self.ctrl = self._new_controller(None, simulated=False)
        self.cam_hw = UEyeCamera()
        self.cam_sim = SimCamera()
        self.cam = self.cam_hw
        self.runner: AutomaticDispatcher | None = None
        self.project_dir: Path | None = None
        self.color_calibration = None
        self.color_calibration_path: Path | None = None
        self.relax_indices: set[int] = set()
        self._active_led_mask = 0
        self._live_stop: threading.Event | None = None

        self._presets: list[dict] = []
        self.led_config: dict = {}
        self._build_ui()
        self._load_config()
        self.load_color_calibration(config.default_color_calibration_path())

    def _load_config(self):
        try:
            self._presets = config.load_presets()
            self.led_config = config.load_led_config()
        except config.ConfigError as exc:
            messagebox.showerror(
                "Configuration", f"{exc}\n\nThe packaged defaults are used instead."
            )
            self._presets = config.load_packaged_json(config.PRESETS_FILE)
            self.led_config = config.load_packaged_json(config.LED_CONFIG_FILE)
        self._refresh_preset_list()
        user_led = config.config_dir() / config.LED_CONFIG_FILE
        if not user_led.is_file():
            self.append_log(
                f"No {config.LED_CONFIG_FILE} in {config.config_dir()}: using the packaged defaults."
            )

    def update_led_mapping(
        self, pol_a: list[int], pol_b: list[int], diffuse_mask: int
    ) -> None:
        merged = config.merge_led_config(self.led_config, pol_a, pol_b, diffuse_mask)
        cleared = (
            "light_azimuths_deg" in self.led_config
            and "light_azimuths_deg" not in merged
        )
        self.led_config = merged
        config.save_led_config(merged)
        self.append_log(f"LED mapping saved: {merged}")
        if cleared:
            self.append_log(
                "WARNING: the set A LEDs changed, so the light azimuths were cleared. "
                "Run 'Calibrate Light Angles' before using photometric stereo."
            )

    def save_light_geometry(
        self, elevation_deg: float, azimuths_deg: list[float]
    ) -> None:
        self.led_config = {
            **self.led_config,
            "light_elevation_deg": elevation_deg,
            "light_azimuths_deg": azimuths_deg,
        }
        config.save_led_config(self.led_config)
        self.append_log(
            f"Light calibration saved: elev={elevation_deg:.1f}°, az={azimuths_deg}"
        )

    def load_color_calibration(self, path: Path | None = None):
        from ...color_calibration import load_bundle

        self.color_calibration = None
        self.color_calibration_path = None
        if path is not None and Path(path).is_file():
            try:
                self.color_calibration = load_bundle(Path(path))
                self.color_calibration_path = Path(path)
                config.update_state(color_calibration=str(Path(path).resolve()))
            except Exception as exc:
                log.warning("Could not read color calibration %s: %s", path, exc)
                self.append_log(f"Could not read color calibration {path}: {exc}")

        if self.color_calibration is None:
            self.keep_raw_var.set(True)
            self.lbl_color_cal.config(
                text="No color calibration loaded: raw images are saved."
            )
        else:
            self.keep_raw_var.set(False)
            exposure = self.color_calibration.diffuse.exposure_ms
            self.lbl_color_cal.config(
                text=f"Color calibration: {self.color_calibration_path.name} ({exposure:g} ms). "
                "Images are corrected when saved."
            )

    def _build_ui(self):
        conn = ttk.LabelFrame(self, text="Connection", padding=6)
        conn.pack(fill="x", padx=8, pady=(8, 2))

        ttk.Label(conn, text="Port:").pack(side="left")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            conn, textvariable=self.port_var, width=18, state="readonly"
        )
        self.port_combo.pack(side="left", padx=4)
        ttk.Button(conn, text="Refresh", command=self._refresh_ports).pack(side="left")
        self.btn_connect = ttk.Button(
            conn, text="Connect", command=self._toggle_connect
        )
        self.btn_connect.pack(side="left", padx=4)

        ttk.Separator(conn, orient="vertical").pack(side="left", fill="y", padx=8)
        self.btn_cam = ttk.Button(conn, text="Open Camera", command=self._toggle_camera)
        self.btn_cam.pack(side="left", padx=4)
        self.sim_camera_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(conn, text="Simulated", variable=self.sim_camera_var).pack(
            side="left"
        )
        self.lbl_cam_status = ttk.Label(conn, text="Camera: closed")
        self.lbl_cam_status.pack(side="left", padx=4)

        ttk.Separator(conn, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(conn, text="Set Project Folder", command=self._choose_project).pack(
            side="left"
        )
        self.lbl_project = ttk.Label(conn, text="(none)")
        self.lbl_project.pack(side="left", padx=4)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)

        meas_frame = ttk.Frame(self.notebook)
        self.notebook.add(meas_frame, text="Measurement")
        self._build_measurement_tab(meas_frame)

        ps_frame = ttk.Frame(self.notebook)
        self.notebook.add(ps_frame, text="Photometric Stereo")
        self.ps_tab = None
        try:
            from .. import stereo

            self.ps_tab = stereo.PhotometricStereoTab(ps_frame, self)
        except ImportError as exc:
            log.exception("Photometric stereo tab unavailable")
            ttk.Label(
                ps_frame,
                text=f"Photometric stereo is unavailable:\n{exc}",
                font=("", 11),
            ).pack(padx=20, pady=40)

        self._refresh_ports()

    def _build_measurement_tab(self, parent):
        paned = ttk.PanedWindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        left = ScrollableFrame(paned, width=400)
        paned.add(left, weight=0)
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._build_controls(left.body)
        self._build_preview(right)

    def _build_controls(self, parent):
        defaults = MeasurementSettings()
        man = ttk.LabelFrame(parent, text="Manual Control", padding=6)
        man.pack(fill="x", padx=4, pady=4)
        self._manual_frame = man

        row = ttk.Frame(man)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Jog steps:").pack(side="left")
        self.jog_var = tk.IntVar(value=200)
        ttk.Entry(row, textvariable=self.jog_var, width=8).pack(side="left", padx=4)
        ttk.Button(row, text="<- Left", command=lambda: self._jog(-1)).pack(side="left")
        ttk.Button(row, text="Right ->", command=lambda: self._jog(1)).pack(
            side="left", padx=2
        )

        row2 = ttk.Frame(man)
        row2.pack(fill="x", pady=2)
        ttk.Button(row2, text="LEDs ON", command=self._leds_on).pack(side="left")
        ttk.Button(row2, text="LEDs OFF", command=self._leds_off).pack(
            side="left", padx=2
        )
        ttk.Button(row2, text="Tension", command=self._read_tension).pack(
            side="left", padx=2
        )
        ttk.Button(row2, text="Tare", command=self._tare).pack(side="left", padx=2)

        row3 = ttk.Frame(man)
        row3.pack(fill="x", pady=2)
        ttk.Button(row3, text="Snap Photo", command=self._snap_photo).pack(side="left")
        ttk.Button(row3, text="Live Preview", command=self._toggle_live).pack(
            side="left", padx=2
        )
        ttk.Button(row3, text="Diffuse ON", command=self._diffuse_on).pack(
            side="left", padx=2
        )
        ttk.Button(row3, text="Diffuse OFF", command=self._diffuse_off).pack(
            side="left", padx=2
        )

        meas = ttk.LabelFrame(parent, text="Measurement Setup", padding=6)
        meas.pack(fill="x", padx=4, pady=4)

        fields = [
            ("Captures:", "count_var", defaults.count, 1, 9999),
            ("Step count:", "stepsize_var", defaults.step_size, 0, 2000000),
            ("LED mask:", "mask_var", defaults.led_mask, 0, 255),
            ("LED power:", "power_var", defaults.led_power, 0, 255),
            ("Settle (ms):", "settle_var", defaults.settle_ms, 0, 5000),
            ("Cam exp (ms):", "camexp_var", defaults.cam_exposure_ms, 0.1, 5000.0),
            ("Release steps:", "release_steps_var", defaults.release_steps, 0, 100000),
        ]
        for label_text, attr, default, lo, hi in fields:
            r = ttk.Frame(meas)
            r.pack(fill="x", pady=1)
            ttk.Label(r, text=label_text, width=14, anchor="e").pack(side="left")
            var = (
                tk.DoubleVar(value=default)
                if isinstance(default, float)
                else tk.IntVar(value=default)
            )
            setattr(self, attr, var)
            ttk.Spinbox(r, from_=lo, to=hi, textvariable=var, width=10).pack(
                side="left", padx=4
            )

        id_frame = ttk.LabelFrame(meas, text="Sample", padding=4)
        id_frame.pack(fill="x", pady=(4, 2))

        r = ttk.Frame(id_frame)
        r.pack(fill="x", pady=1)
        ttk.Label(r, text="Sample ID:", width=14, anchor="e").pack(side="left")
        self.sample_id_var = tk.StringVar(value=defaults.sample_id)
        ttk.Entry(r, textvariable=self.sample_id_var, width=16).pack(
            side="left", padx=4
        )

        r2 = ttk.Frame(id_frame)
        r2.pack(fill="x", pady=1)
        ttk.Label(r2, text="Direction:", width=14, anchor="e").pack(side="left")
        self.direction_var = tk.StringVar(value=defaults.direction)
        for val, label_text in [("warp", "Warp"), ("weft", "Weft"), ("bias", "Bias")]:
            ttk.Radiobutton(
                r2, text=label_text, variable=self.direction_var, value=val
            ).pack(side="left", padx=4)

        r3 = ttk.Frame(id_frame)
        r3.pack(fill="x", pady=1)
        ttk.Label(r3, text="Motion:", width=14, anchor="e").pack(side="left")
        self.motion_mode_var = tk.StringVar(value=defaults.motion_mode)
        for val, label_text in [("loading", "Loading"), ("release", "Release")]:
            ttk.Radiobutton(
                r3, text=label_text, variable=self.motion_mode_var, value=val
            ).pack(side="left", padx=4)

        for label_text, attr, value, hi, inc, note in [
            (
                "Clamp width (m):",
                "clamp_width_var",
                defaults.clamp_width_m,
                10.0,
                0.001,
                "",
            ),
            (
                "Max stretch (mm):",
                "max_stretch_var",
                defaults.max_stretch_mm,
                100000.0,
                0.1,
                "0 = off",
            ),
            (
                "Max force (N):",
                "max_force_var",
                defaults.max_force_n,
                10000.0,
                0.5,
                "0 = off",
            ),
        ]:
            r = ttk.Frame(id_frame)
            r.pack(fill="x", pady=1)
            ttk.Label(r, text=label_text, width=15, anchor="e").pack(side="left")
            var = tk.DoubleVar(value=value)
            setattr(self, attr, var)
            ttk.Spinbox(
                r,
                from_=0.001 if attr == "clamp_width_var" else 0.0,
                to=hi,
                increment=inc,
                textvariable=var,
                width=10,
            ).pack(side="left", padx=4)
            if note:
                ttk.Label(r, text=note).pack(side="left")

        self.speed_var = tk.IntVar(value=defaults.speed_mult)
        self._speed_btn = ttk.Button(meas, text="Speed: 1×", command=self._toggle_speed)
        self._speed_btn.pack(fill="x", pady=(2, 4))

        mode_frame = ttk.LabelFrame(meas, text="Capture Mode", padding=4)
        mode_frame.pack(fill="x", pady=(4, 2))
        self.mode_var = tk.StringVar(value=defaults.capture_mode)
        ttk.Radiobutton(
            mode_frame,
            text="RGB",
            variable=self.mode_var,
            value="rgb",
            command=self._on_mode_change,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame,
            text="Sequential white LEDs",
            variable=self.mode_var,
            value="sequential",
            command=self._on_mode_change,
        ).pack(anchor="w")
        self.polarization_var = tk.BooleanVar(value=defaults.polarization)
        self.chk_pol = ttk.Checkbutton(
            mode_frame, text="+ Polarization", variable=self.polarization_var
        )
        self.chk_pol.pack(anchor="w", padx=18)
        self.include_diffuse_var = tk.BooleanVar(value=defaults.include_diffuse)
        self.chk_diff = ttk.Checkbutton(
            mode_frame,
            text="+ Diffuse reference shot",
            variable=self.include_diffuse_var,
        )
        self.chk_diff.pack(anchor="w", padx=18)

        self._dialog_buttons = [
            ttk.Button(
                mode_frame, text="Identify LEDs", command=self._open_led_identifier
            ),
            ttk.Button(
                mode_frame,
                text="Calibrate Light Angles",
                command=self._open_light_calibration,
            ),
            ttk.Button(
                mode_frame,
                text="Calibrate Colors",
                command=self._open_color_calibration,
            ),
        ]
        for i, button in enumerate(self._dialog_buttons):
            button.pack(anchor="w", pady=(4, 0) if i == 0 else (2, 0))
        self.lbl_color_cal = ttk.Label(
            mode_frame, text="", wraplength=260, justify="left"
        )
        self.lbl_color_cal.pack(anchor="w", padx=18, pady=(2, 0))
        self.keep_raw_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            mode_frame,
            text="Keep uncalibrated (raw) images",
            variable=self.keep_raw_var,
        ).pack(anchor="w", padx=18)

        relax_frame = ttk.LabelFrame(meas, text="Relaxation Capture", padding=4)
        relax_frame.pack(fill="x", pady=(4, 2))
        self.relax_enabled_var = tk.BooleanVar(value=defaults.relax_enabled)
        ttk.Checkbutton(
            relax_frame,
            text="Enable relaxation capture",
            variable=self.relax_enabled_var,
        ).pack(anchor="w")
        rr1 = ttk.Frame(relax_frame)
        rr1.pack(fill="x", pady=1)
        ttk.Button(
            rr1, text="Select Relax Points", command=self._open_relax_dialog
        ).pack(side="left", padx=4)
        self.lbl_relax_pts = ttk.Label(rr1, text="Selected: None")
        self.lbl_relax_pts.pack(side="left", padx=4)
        rr2 = ttk.Frame(relax_frame)
        rr2.pack(fill="x", pady=1)
        ttk.Label(rr2, text="Duration (ms):", width=14, anchor="e").pack(side="left")
        self.relax_duration_var = tk.IntVar(value=defaults.relax_duration_ms)
        ttk.Spinbox(
            rr2, from_=100, to=600000, textvariable=self.relax_duration_var, width=10
        ).pack(side="left", padx=4)

        btn_row = ttk.Frame(meas)
        btn_row.pack(fill="x", pady=6)
        self.btn_run = ttk.Button(
            btn_row, text=">  Run Measurement", command=self._run_measurement
        )
        self.btn_run.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_stop = ttk.Button(
            btn_row, text="X  Stop", command=self._stop_measurement, state="disabled"
        )
        self.btn_stop.pack(side="left", padx=2)

        self.progress = ttk.Progressbar(meas, mode="determinate")
        self.progress.pack(fill="x", pady=2)
        self.lbl_progress = ttk.Label(meas, text="Idle")
        self.lbl_progress.pack()

        pre = ttk.LabelFrame(parent, text="Presets", padding=6)
        pre.pack(fill="x", padx=4, pady=4)
        self._preset_listbox = tk.Listbox(
            pre, height=5, selectmode="browse", activestyle="dotbox"
        )
        self._preset_listbox.pack(fill="x")
        self._preset_listbox.bind("<Double-Button-1>", lambda _: self._load_preset())
        pb = ttk.Frame(pre)
        pb.pack(fill="x", pady=(4, 0))
        ttk.Button(pb, text="Load", command=self._load_preset).pack(
            side="left", expand=True, fill="x", padx=1
        )
        ttk.Button(pb, text="Save", command=self._save_preset).pack(
            side="left", expand=True, fill="x", padx=1
        )
        ttk.Button(pb, text="Delete", command=self._delete_preset).pack(
            side="left", expand=True, fill="x", padx=1
        )

        rst = ttk.Frame(parent, padding=4)
        rst.pack(fill="x", padx=4, pady=4)
        ttk.Button(
            rst, text="Reset All to Defaults", command=self._reset_defaults
        ).pack(fill="x")

        self._on_mode_change()

    def _build_preview(self, parent):
        self.canvas = ImageCanvas(parent, bg="#1e1e1e")
        self.canvas.pack(fill="both", expand=True, pady=(0, 4))

        self.log_text = scrolledtext.ScrolledText(
            parent,
            height=10,
            state="disabled",
            font=("Consolas", 9),
            bg="#111",
            fg="#ccc",
        )
        self.log_text.pack(fill="x")

        raw = ttk.Frame(parent)
        raw.pack(fill="x", pady=(2, 0))
        ttk.Label(raw, text="Serial:").pack(side="left")
        self.raw_var = tk.StringVar()
        entry = ttk.Entry(raw, textvariable=self.raw_var)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        entry.bind("<Return>", lambda _: self._send_raw())
        self._raw_entry = entry
        self._raw_send = ttk.Button(raw, text="Send", command=self._send_raw)
        self._raw_send.pack(side="left")

    def _on_mode_change(self):
        state = "normal" if self.mode_var.get() == "sequential" else "disabled"
        self.chk_pol.config(state=state)
        self.chk_diff.config(state=state)

    def read_settings(self, validate: bool = True) -> MeasurementSettings:
        try:
            settings = MeasurementSettings(
                count=self.count_var.get(),
                step_size=abs(self.stepsize_var.get()),
                motion_mode=self.motion_mode_var.get(),
                led_mask=self.mask_var.get(),
                led_power=self.power_var.get(),
                settle_ms=self.settle_var.get(),
                cam_exposure_ms=self.camexp_var.get(),
                sample_id=self.sample_id_var.get().strip() or "sample",
                direction=self.direction_var.get(),
                clamp_width_m=self.clamp_width_var.get(),
                max_stretch_mm=self.max_stretch_var.get(),
                max_force_n=self.max_force_var.get(),
                speed_mult=self.speed_var.get(),
                relax_enabled=self.relax_enabled_var.get(),
                relax_indices=sorted(self.relax_indices),
                relax_duration_ms=self.relax_duration_var.get(),
                capture_mode=self.mode_var.get(),
                polarization=self.polarization_var.get(),
                include_diffuse=self.include_diffuse_var.get(),
                release_steps=self.release_steps_var.get(),
            )
        except tk.TclError as exc:
            raise ValueError(
                f"A setting field is empty or not a number ({exc})."
            ) from exc
        if validate:
            settings.validate()

        return settings

    def _apply_settings(self, s: MeasurementSettings):
        self.count_var.set(s.count)
        self.stepsize_var.set(s.step_size)
        self.motion_mode_var.set(s.motion_mode)
        self.mask_var.set(s.led_mask)
        self.power_var.set(s.led_power)
        self.settle_var.set(s.settle_ms)
        self.camexp_var.set(s.cam_exposure_ms)
        self.sample_id_var.set(s.sample_id)
        self.direction_var.set(s.direction)
        self.clamp_width_var.set(s.clamp_width_m)
        self.max_stretch_var.set(s.max_stretch_mm)
        self.max_force_var.set(s.max_force_n)
        self.speed_var.set(s.speed_mult)
        self._speed_btn.config(text=f"Speed: {s.speed_mult}×")
        self.relax_enabled_var.set(s.relax_enabled)
        self.relax_indices = set(s.relax_indices)
        self._update_relax_label()
        self.relax_duration_var.set(s.relax_duration_ms)
        self.mode_var.set(s.capture_mode)
        self.polarization_var.set(s.polarization)
        self.include_diffuse_var.set(s.include_diffuse)
        self.release_steps_var.set(s.release_steps)
        self._on_mode_change()

    def _load_preset(self):
        sel = self._preset_listbox.curselection()
        if not sel:
            return
        preset = self._presets[sel[0]]
        try:
            current = self.read_settings(validate=False)
        except ValueError:
            current = None  # an unreadable field is overwritten by the preset anyway
        self._apply_settings(MeasurementSettings.from_preset(preset, current))
        self.append_log(f"Preset loaded: {preset.get('name', '?')}")

    def _save_preset(self):
        name = simpledialog.askstring("Save Preset", "Preset name:", parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        try:
            settings = self.read_settings(validate=False).to_preset(name)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))

            return
        for i, preset in enumerate(self._presets):
            if preset.get("name") == name:
                if not messagebox.askyesno("Overwrite?", f'Overwrite preset "{name}"?'):
                    return
                self._presets[i] = settings
                break
        else:
            self._presets.append(settings)
        self._store_presets()
        self.append_log(f"Preset saved: {name}")

    def _delete_preset(self):
        sel = self._preset_listbox.curselection()
        if not sel:
            return
        name = self._presets[sel[0]].get("name", "?")
        if not messagebox.askyesno("Delete preset", f'Delete "{name}"?'):
            return
        self._presets.pop(sel[0])
        self._store_presets()
        self.append_log(f"Preset deleted: {name}")

    def _store_presets(self):
        try:
            path = config.save_presets(self._presets)
            self.append_log(f"Presets written to {path}")
        except OSError as exc:
            messagebox.showerror("Presets", f"Could not write the presets file: {exc}")
        self._refresh_preset_list()

    def _refresh_preset_list(self):
        self._preset_listbox.delete(0, "end")
        for preset in self._presets:
            self._preset_listbox.insert("end", preset.get("name", "(unnamed)"))

    def _reset_defaults(self):
        self._apply_settings(MeasurementSettings())
        self.append_log("Defaults restored")

    def _update_relax_label(self):
        if not self.relax_indices:
            self.lbl_relax_pts.config(text="Selected: None")

            return
        text = ", ".join(str(i) for i in sorted(self.relax_indices))
        self.lbl_relax_pts.config(
            text=f"Selected: {text if len(text) <= 30 else text[:27] + '...'}"
        )

    def _open_relax_dialog(self):
        try:
            total = self.count_var.get()
        except tk.TclError:
            return
        if total > 0:
            RelaxPointsDialog(self, total, self.relax_indices, self._set_relax_indices)

    def _set_relax_indices(self, indices: set[int]):
        self.relax_indices = indices
        self._update_relax_label()

    def _refresh_ports(self):
        try:
            ports = ArduinoController.list_ports()
        except Exception as exc:
            log.warning("Could not list serial ports: %s", exc)
            ports = []
        ports.append(SIMULATED_PORT)
        self.port_combo["values"] = ports
        if ports:
            self.port_var.set(ports[0])

    def _toggle_connect(self):
        if self.ctrl.connected:
            self.ctrl.disconnect()
            self.btn_connect.config(text="Connect")
            self.append_log("Disconnected")

            return
        name = self.port_var.get()
        if not name:
            messagebox.showwarning("No port", "Select a serial port first.")

            return
        simulated = name == SIMULATED_PORT
        self.ctrl = self._new_controller(FakePort if simulated else None, simulated)
        self.btn_connect.config(state="disabled")
        self.append_log(f"Connecting to {name} ...")

        def connect():
            self.ctrl.connect(name, settle_s=0.0 if simulated else 2.0)

        def done(_result, error):
            self.btn_connect.config(state="normal")
            if error is not None:
                messagebox.showerror("Connection Error", str(error))

                return
            self.btn_connect.config(text="Disconnect")
            self.append_log(f"Connected to {name}")

        self.run_async(connect, done)

    def _new_controller(self, port_factory, simulated: bool) -> ArduinoController:
        ctrl = ArduinoController(port_factory)
        ctrl.simulated = simulated
        ctrl.add_listener(lambda line: self.post(self.append_log, f"<- {line}"))

        return ctrl

    def _toggle_camera(self):
        if self.cam.is_open:
            self._stop_live()
            self.cam.close()
            self.lbl_cam_status.config(text="Camera: closed")
            self.btn_cam.config(text="Open Camera")

            return
        self.cam = self.cam_sim if self.sim_camera_var.get() else self.cam_hw
        try:
            self.cam.open()
        except Exception as exc:
            messagebox.showerror("Camera Error", str(exc))

            return
        width, height = self.cam.size
        kind = " (simulated)" if self.cam.simulated else ""
        self.lbl_cam_status.config(text=f"Camera: {width}x{height}{kind}")
        self.btn_cam.config(text="Close Camera")
        self.append_log(f"Camera opened{kind}")

    def _choose_project(self):
        remembered = config.load_state().get("project_dir")
        d = filedialog.askdirectory(
            title="Select Project Folder", initialdir=remembered or str(Path.home())
        )
        if d:
            self.project_dir = Path(d)
            config.update_state(project_dir=d)
            self.lbl_project.config(text=str(self.project_dir))
            self.append_log(f"Project folder: {self.project_dir}")

    def _require_serial(self) -> bool:
        if not self.ctrl.connected:
            messagebox.showwarning("Not connected", "Connect to the controller first.")

            return False

        return True

    def _jog(self, direction: int):
        if self._require_serial():
            self.ctrl.jog(direction * self.jog_var.get())

    @property
    def _diffuse_mask(self) -> int:
        return int(self.led_config.get("diffuse_mask", 0))

    def _leds_on(self):
        if self._require_serial():
            directional = self.mask_var.get() & ~self._diffuse_mask
            self._active_led_mask = directional | (
                self._active_led_mask & self._diffuse_mask
            )
            self.ctrl.set_power(self.power_var.get())
            self.ctrl.set_leds(self._active_led_mask)

    def _leds_off(self):
        if self._require_serial():
            self._active_led_mask &= self._diffuse_mask
            self.ctrl.set_leds(self._active_led_mask)
            if self._active_led_mask == 0:
                self.ctrl.set_power(0)

    def _diffuse_on(self):
        if self._require_serial():
            if self._diffuse_mask == 0:
                messagebox.showwarning("No diffuse mask", "Map the diffuse light first")

                return
            self._active_led_mask |= self._diffuse_mask
            self.ctrl.set_power(self.power_var.get())
            self.ctrl.set_leds(self._active_led_mask)

    def _diffuse_off(self):
        if self._require_serial():
            self._active_led_mask &= ~self._diffuse_mask
            self.ctrl.set_leds(self._active_led_mask)
            if self._active_led_mask == 0:
                self.ctrl.set_power(0)

    def _read_tension(self):
        if self._require_serial():
            self.ctrl.read_tension()

    def _tare(self):
        if self._require_serial():
            self.ctrl.tare_tension()

    def _toggle_speed(self):
        if self._run_active:
            return  # the run's move deadlines are computed for the speed it started with
        new = 2 if self.speed_var.get() == 1 else 1
        self.speed_var.set(new)
        self._speed_btn.config(text=f"Speed: {new}×")
        if self.ctrl.connected:
            self.ctrl.set_speed(new)

    @property
    def _run_active(self) -> bool:
        return self.runner is not None and self.runner.running

    def _set_run_locked(self, locked: bool):
        flag = ["disabled"] if locked else ["!disabled"]
        widgets = [
            self.btn_cam,
            self.btn_connect,
            self._speed_btn,
            self._raw_entry,
            self._raw_send,
            *self._dialog_buttons,
        ]
        for row in self._manual_frame.winfo_children():
            widgets.extend(row.winfo_children())
        for widget in widgets:
            widget.state(flag)

    def _show_image(self, image: Image.Image):
        self.canvas.show_image(image)

    def _snap_photo(self):
        if not self.cam.is_open:
            self.append_log("Camera not open: nothing to snap")

            return

        def snap():
            frame = self.cam.capture_frame()

            return None if frame is None else frame_to_pil(frame)

        def done(image, error):
            if error is not None or image is None:
                self.append_log(f"Snap failed: {error or 'no frame'}")

                return
            self._show_image(image)
            if self.project_dir:
                path = (
                    self.project_dir
                    / f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                image.save(path)
                self.append_log(f"Snap saved: {path.name}")

        self.run_async(snap, done)

    def _toggle_live(self):
        if self._live_stop is not None:
            self._stop_live()

            return
        if not self.cam.is_open:
            self.append_log("Open camera first")

            return
        stop = self._live_stop = threading.Event()
        shown = threading.Event()
        shown.set()

        def show(image):
            self._show_image(image)
            shown.set()

        def worker():
            while not stop.is_set() and self.cam.is_open:
                if shown.wait(0.1):
                    frame = self.cam.capture_frame()
                    if frame is not None:
                        shown.clear()
                        self.post(show, frame_to_pil(frame))
                time.sleep(0.03)

        threading.Thread(target=worker, name="live-preview", daemon=True).start()

    def _stop_live(self):
        if self._live_stop is not None:
            self._live_stop.set()
            self._live_stop = None

    def _run_measurement(self):
        if not self._require_serial():
            return
        if self.project_dir is None:
            messagebox.showwarning(
                "No project folder", "Choose a project folder first."
            )

            return
        if self._run_active:
            return
        if not self.cam.is_open:
            messagebox.showwarning(
                "Camera closed", "Open the camera (or the simulated camera) first."
            )

            return
        try:
            settings = self.read_settings()
            plan = session.build_capture_plan(settings, self.led_config)
        except ValueError as exc:
            messagebox.showerror("Cannot start", str(exc))

            return

        if settings.cam_exposure_ms >= EXPOSURE_WARN_MS:
            self.append_log(
                f"WARNING: camera exposure is {EXPOSURE_WARN_MS:g} ms or higher; "
                "the sensor is reported to clip there."
            )
        if settings.capture_mode == "rgb":
            self.append_log(
                "NOTE: RGB runs cannot be solved by the Photometric Stereo tab; "
                "use Sequential white LEDs for photometric stereo."
            )
        base = session.run_dir_for(self.project_dir, settings)
        run_dir = session.next_free_run_dir(base)
        if run_dir != base and not messagebox.askyesno(
            "Run exists",
            f"{base.name} already holds a measurement.\nSave this run as {run_dir.name} instead?\n\n"
            f"The analysis tools only find runs named <direction>_<loading|release>: rename "
            f"the folder before processing it.",
        ):
            return

        self._stop_live()
        self.runner = AutomaticDispatcher(
            self.ctrl,
            self.cam,
            run_dir,
            settings,
            plan,
            dict(self.led_config),
            color_bundle=self.color_calibration,
            keep_raw=self.keep_raw_var.get(),
            on_image=lambda img: self.post(self._show_image, img),
            on_progress=lambda cur, tot: self.post(self._update_progress, cur, tot),
            on_done=lambda status, reason: self.post(
                self._measurement_done, status, reason
            ),
            on_log=lambda msg: self.post(self.append_log, msg),
        )
        try:
            self.runner.start()
        except Exception as exc:
            log.exception("Could not start the measurement")
            messagebox.showerror("Cannot start", str(exc))
            self.runner = None

            return

        self.progress["maximum"] = settings.count
        self.progress["value"] = 0
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._set_run_locked(True)
        self.lbl_progress.config(text=f"Running 0/{settings.count}")
        self.append_log(f"Run folder: {run_dir}")

    def _stop_measurement(self):
        if self.runner:
            self.runner.cancel(reason="user_stop")

    def _update_progress(self, current: int, total: int):
        self.progress["value"] = current
        self.lbl_progress.config(text=f"Running {current}/{total}")

    def _measurement_done(self, status: str, reason: str | None):
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self._set_run_locked(False)
        self._active_led_mask = 0
        if status == "complete":
            self.lbl_progress.config(text="Complete")
        else:
            self.lbl_progress.config(
                text=f"Aborted: {reason}" if reason else "Cancelled"
            )

    def _send_raw(self):
        cmd = self.raw_var.get().strip()
        if cmd and self.ctrl.connected and not self._run_active:
            self.ctrl.send(cmd)
            self.append_log(f"-> {cmd}")
            self.raw_var.set("")

    def append_log(self, text: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _open_led_identifier(self):
        if self._require_serial():
            LedIdentifierDialog(self)

    def _open_color_calibration(self):
        ColorCalibrationDialog(self)

    def _open_light_calibration(self):
        if not self._require_serial():
            return
        if not self.cam.is_open:
            messagebox.showwarning("Camera required", "Open the camera first.")

            return
        LightCalibrationDialog(self)

    def _on_close(self):
        self._stop_live()
        if self._run_active:
            self.runner.cancel(reason="window_close")
            self.runner.join(timeout=5.0)
        self.cam_hw.close()
        self.cam_sim.close()
        self.ctrl.disconnect()
        self.destroy()
