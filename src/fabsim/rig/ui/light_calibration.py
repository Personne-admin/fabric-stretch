from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image

from ..hardware.camera import frame_to_pil
from .widgets import ImageCanvas


def shadow_to_light_azimuth(
    base: tuple[float, float], tip: tuple[float, float]
) -> float:
    shadow_dx = tip[0] - base[0]
    shadow_dy_up = -(tip[1] - base[1])  # image y points down
    shadow_az = float(np.degrees(np.arctan2(shadow_dy_up, shadow_dx)))

    return (shadow_az + 180.0) % 360.0


class LightCalibrationDialog:
    N_LEDS = 3

    def __init__(self, app):
        self.app = app
        self.ctrl = app.ctrl
        self.cam = app.cam
        self.led_cfg = app.led_config
        self.pol_a_bits = list(self.led_cfg.get("pol_a_bits", []))
        if len(self.pol_a_bits) < self.N_LEDS:
            messagebox.showwarning(
                "LED mapping incomplete",
                "Calibration needs 3 directional LEDs assigned.",
            )

            return

        self.current_idx = 0
        self.measurements: list[dict] = []
        self.click_state = "idle"
        self.current_base: tuple[float, float] | None = None
        self._build()

    def _build(self):
        dlg = tk.Toplevel(self.app)
        self.dlg = dlg
        dlg.title("Light Azimuth Calibration")
        dlg.geometry("900x720")
        dlg.transient(self.app)
        dlg.grab_set()
        dlg.protocol("WM_DELETE_WINDOW", self._cancel)

        ttk.Label(
            dlg,
            text=(
                "Place a small upright object where the camera can see both it AND its shadow"
                " on a roughly flat surface. For each LED, click Capture, then click the BASE"
                " of the object, then the TIP of its shadow."
            ),
            wraplength=860,
            justify="left",
        ).pack(padx=10, pady=(10, 4), anchor="w")

        params = ttk.Frame(dlg)
        params.pack(fill="x", padx=10, pady=4)
        ttk.Label(params, text="LED power:").pack(side="left")
        self.power_var = tk.IntVar(value=self.app.power_var.get())
        ttk.Spinbox(
            params, from_=10, to=255, textvariable=self.power_var, width=6
        ).pack(side="left", padx=4)
        ttk.Separator(params, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(params, text="Elevation (°):").pack(side="left")
        self.elev_var = tk.DoubleVar(
            value=self.led_cfg.get("light_elevation_deg", 45.0)
        )
        ttk.Spinbox(
            params,
            from_=10.0,
            to=85.0,
            increment=0.5,
            textvariable=self.elev_var,
            width=6,
        ).pack(side="left", padx=4)
        ttk.Label(params, text="(physical tilt above horizontal)").pack(
            side="left", padx=4
        )

        self.status_var = tk.StringVar(
            value=f"Click 'Capture LED 1 (bit {self.pol_a_bits[0]})' to begin."
        )
        ttk.Label(dlg, textvariable=self.status_var, font=("", 11, "bold")).pack(
            padx=10, pady=4, anchor="w"
        )

        self.canvas = ImageCanvas(
            dlg,
            bg="#202020",
            height=400,
            on_click=self._on_click,
            on_render=self._draw_overlays,
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=4)

        res_frame = ttk.LabelFrame(dlg, text="Results", padding=6)
        res_frame.pack(fill="x", padx=10, pady=4)
        self.results_var = tk.StringVar(value="(no measurements yet)")
        ttk.Label(
            res_frame,
            textvariable=self.results_var,
            font=("Consolas", 10),
            justify="left",
        ).pack(anchor="w")

        btns = ttk.Frame(dlg)
        btns.pack(fill="x", padx=10, pady=8)
        self.btn_capture = ttk.Button(
            btns,
            text=f"Capture LED 1 (bit {self.pol_a_bits[0]})",
            command=self._capture,
        )
        self.btn_capture.pack(side="left", padx=4)
        self.btn_redo = ttk.Button(
            btns, text="Redo This LED", command=self._redo, state="disabled"
        )
        self.btn_redo.pack(side="left", padx=4)
        ttk.Frame(btns).pack(side="left", fill="x", expand=True)
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side="right", padx=4)
        self.btn_save = ttk.Button(
            btns, text="Save Calibration", command=self._save, state="disabled"
        )
        self.btn_save.pack(side="right", padx=4)

    def _shoot(self, mask: int, power: int):
        with self.ctrl.lights(mask, power):
            frame = self.cam.capture_frame()

        return None if frame is None else frame_to_pil(frame)

    def _capture(self):
        if not self.cam.is_open:
            messagebox.showerror("Camera closed", "Open the camera first.")

            return
        if not self.ctrl.connected:
            messagebox.showerror("Not connected", "Connect to the controller first.")

            return

        bit = self.pol_a_bits[self.current_idx]
        self.status_var.set(f"Capturing with LED {self.current_idx + 1} (bit {bit})...")
        self.btn_capture.config(state="disabled")
        self.btn_redo.config(state="disabled")
        self.app.run_async(
            lambda: self._shoot(1 << bit, self.power_var.get()), self._captured
        )

    def _captured(self, image: Image.Image | None, error: Exception | None):
        if error is not None or image is None:
            messagebox.showerror(
                "Capture failed",
                str(error) if error else "No frame returned from camera.",
            )
            self.status_var.set(f"Capture failed, retry LED {self.current_idx + 1}")
            self.btn_capture.config(state="normal")

            return
        self.current_base = None
        self.click_state = "wait_base"
        self.measurements = [
            m for m in self.measurements if m["led_idx"] != self.current_idx
        ]
        self._update_results()
        self.canvas.show_image(image)
        self.status_var.set(
            f"LED {self.current_idx + 1}: click the BASE of the upright object."
        )
        self.btn_redo.config(state="normal")

    def _on_click(self, ix: float, iy: float):
        if self.click_state == "wait_base":
            self.current_base = (ix, iy)
            self.click_state = "wait_tip"
            self.status_var.set(
                f"LED {self.current_idx + 1}: now click the TIP of the SHADOW."
            )
            self.canvas.show_image(self.canvas.image)
        elif self.click_state == "wait_tip":
            azimuth = shadow_to_light_azimuth(self.current_base, (ix, iy))
            self.measurements.append(
                {
                    "led_idx": self.current_idx,
                    "bit": self.pol_a_bits[self.current_idx],
                    "base": self.current_base,
                    "tip": (ix, iy),
                    "azimuth_deg": azimuth,
                }
            )
            self.click_state = "done"
            self.canvas.show_image(self.canvas.image)
            self.status_var.set(
                f"LED {self.current_idx + 1}: azimuth ≈ {azimuth:.1f}°."
            )
            self._update_results()
            self._advance_or_finish()

    def _advance_or_finish(self):
        if self.current_idx + 1 < self.N_LEDS:
            self.current_idx += 1
            self.click_state = "idle"
            self.btn_capture.config(
                state="normal",
                text=f"Capture LED {self.current_idx + 1} (bit {self.pol_a_bits[self.current_idx]})",
            )
            self.btn_redo.config(state="disabled")
            self.status_var.set(
                f"Click 'Capture LED {self.current_idx + 1}' when the next LED is ready."
            )
        else:
            self.btn_capture.config(state="disabled", text="All 3 LEDs done")
            self.btn_save.config(state="normal")
            self.status_var.set("All 3 LEDs measured")

    def _redo(self):
        self.measurements = [
            m for m in self.measurements if m["led_idx"] != self.current_idx
        ]
        self.btn_save.config(state="disabled")
        self._update_results()
        self._capture()

    def _draw_overlays(self, canvas: ImageCanvas):
        for m in self.measurements:
            if m["led_idx"] != self.current_idx:
                self._draw_arrow(
                    m["base"],
                    m["tip"],
                    "#777",
                    f"L{m['led_idx'] + 1}: {m['azimuth_deg']:.0f}°",
                )
        if self.current_base is not None and self.click_state in ("wait_tip", "done"):
            self._draw_marker(self.current_base, "#ff5050", "BASE")
        current = next(
            (m for m in self.measurements if m["led_idx"] == self.current_idx), None
        )
        if current is not None:
            self._draw_arrow(current["base"], current["tip"], "#ffff60", "shadow")
            bx, by = current["base"]
            tx, ty = current["tip"]
            self._draw_arrow(
                current["base"],
                (bx - (tx - bx), by - (ty - by)),
                "#60ff60",
                f"-> LED {self.current_idx + 1}",
            )

    def _draw_marker(self, img_pt, color, label):
        cx, cy = self.canvas.image_to_canvas(img_pt)
        r = 7
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
        if label:
            self.canvas.create_text(
                cx + 14,
                cy - 14,
                text=label,
                fill=color,
                font=("", 10, "bold"),
                anchor="w",
            )

    def _draw_arrow(self, p_from, p_to, color, label):
        a = self.canvas.image_to_canvas(p_from)
        b = self.canvas.image_to_canvas(p_to)
        self.canvas.create_line(
            *a, *b, fill=color, width=2, arrow="last", arrowshape=(12, 14, 5)
        )
        if label:
            self.canvas.create_text(
                (a[0] + b[0]) / 2 + 8,
                (a[1] + b[1]) / 2,
                text=label,
                fill=color,
                font=("", 9),
                anchor="w",
            )

    def _update_results(self):
        if not self.measurements:
            self.results_var.set("(no measurements yet)")

            return
        lines = [
            f"LED {m['led_idx'] + 1} (bit {m['bit']}): azimuth = {m['azimuth_deg']:7.2f}°"
            for m in sorted(self.measurements, key=lambda x: x["led_idx"])
        ]
        if len(self.measurements) == self.N_LEDS:
            azs = sorted(m["azimuth_deg"] for m in self.measurements)
            diffs = [(azs[(i + 1) % 3] - azs[i]) % 360 for i in range(3)]
            lines += [
                "",
                "Spacing between adjacent LEDs: "
                f"{diffs[0]:.1f}°, {diffs[1]:.1f}°, {diffs[2]:.1f}°  (ideal: 120°)",
            ]
            max_dev = max(abs(d - 120) for d in diffs)
            if max_dev > 20:
                lines.append(
                    f"  WARNING: deviation up to {max_dev:.1f}° from 120°. Likely error, consider redoing."
                )
            else:
                lines.append("  Spacing looks consistent with a symmetric rig.")
        self.results_var.set("\n".join(lines))

    def _save(self):
        if len(self.measurements) != self.N_LEDS:
            return
        ordered = sorted(self.measurements, key=lambda x: x["led_idx"])
        azimuths = [round(m["azimuth_deg"], 2) for m in ordered]
        self.app.save_light_geometry(float(self.elev_var.get()), azimuths)
        self._cleanup()

    def _cancel(self):
        self._cleanup()

    def _cleanup(self):
        if self.ctrl.connected:
            try:
                self.ctrl.lights_off()
            except Exception:
                pass
        self.dlg.destroy()
