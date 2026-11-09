import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from ...color_calibration import ColorCalibration, apply_model, fit_model, save_bundle
from ..colorcorrect import basenames
from ..hardware.camera import frame_to_pil
from ..settings import EXPOSURE_TOLERANCE


class ColorCalibrationDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.title("Color calibration · 30 patch chart")
        self.geometry("1000x760")
        self.app = app
        self.led_bits = list(app.led_config.get("pol_a_bits", []))
        self.images = {}
        self.points = {}
        self.models = {}
        self.role = tk.StringVar(value="diffuse")
        self.led = tk.IntVar(value=0)
        self.exposure = tk.DoubleVar(value=app.camexp_var.get())
        self.status = tk.StringVar(
            value="Load a chart image, then mark its four OUTER corners: "
            "top left, top right, bottom right, bottom left."
        )
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")
        ttk.Label(bar, text="Image type:").pack(side="left")
        for label, value in (("Diffuse", "diffuse"), ("Point light", "point_light")):
            ttk.Radiobutton(bar, text=label, variable=self.role, value=value, command=self._refresh).pack(
                side="left", padx=5
            )
        ttk.Button(bar, text="Open chart image...", command=self._open).pack(side="left", padx=12)
        ttk.Button(bar, text="Use camera frame", command=self._capture).pack(side="left")
        ttk.Label(bar, text="Exposure (ms):").pack(side="left", padx=(18, 3))
        ttk.Entry(bar, textvariable=self.exposure, width=8).pack(side="left")
        leds = ttk.Frame(self, padding=(8, 0, 8, 8))
        leds.pack(fill="x")
        ttk.Label(leds, text="Point light LEDs:").pack(side="left")
        for index, bit in enumerate(self.led_bits):
            ttk.Button(
                leds,
                text=f"LED {index + 1} (bit {bit})",
                command=lambda i=index, b=bit: self._led_on(i, b),
            ).pack(side="left", padx=4)
        ttk.Button(leds, text="Diffuse light", command=self._diffuse_on).pack(side="left", padx=(12, 0))
        ttk.Button(leds, text="All LEDs off", command=self._leds_off).pack(side="left", padx=12)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.canvas = tk.Canvas(self, bg="#23262b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Configure>", lambda _event: self._refresh())
        ttk.Label(self, textvariable=self.status, wraplength=950).pack(fill="x", padx=10, pady=5)
        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Reset corners", command=self._reset).pack(side="left")
        ttk.Button(bottom, text="Fit this image", command=self._fit).pack(side="left", padx=8)
        ttk.Button(bottom, text="Preview correction", command=self._preview).pack(side="left")
        ttk.Button(bottom, text="Save both models...", command=self._save).pack(side="right")

    def _key(self):
        # Each point light LED gets its own image slot; diffuse has one.
        if self.role.get() == "diffuse":
            return "diffuse"

        return f"point_light/{self.led.get()}"

    def _point_keys(self):
        return [f"point_light/{i}" for i in range(len(self.led_bits))]

    def _label(self):
        if self.role.get() == "diffuse":
            return "diffuse"
        i = self.led.get()

        return f"point light {i + 1} of {len(self.led_bits)} (LED {i + 1}, bit {self.led_bits[i]})"

    def _next_point(self):
        for i, key in enumerate(self._point_keys()):
            if key not in self.images:
                self.led.set(i)

                return

    def _led_on(self, index, bit):
        self.role.set("point_light")
        self.led.set(index)
        self._refresh()
        if not self.app.ctrl.connected:
            messagebox.showwarning("Not connected", "Connect to the controller first.", parent=self)

            return
        self.app.ctrl.set_power(self.app.power_var.get())
        self.app.ctrl.set_leds(1 << bit)

    def _diffuse_on(self):
        self.role.set("diffuse")
        self._refresh()
        mask = int(self.app.led_config.get("diffuse_mask", 0))
        if not self.app.ctrl.connected:
            messagebox.showwarning("Not connected", "Connect to the controller first.", parent=self)
        elif mask == 0:
            messagebox.showwarning(
                "No diffuse mask",
                "Map the diffuse light first (Identify LEDs).",
                parent=self,
            )
        else:
            self.app.ctrl.set_power(self.app.power_var.get())
            self.app.ctrl.set_leds(mask)

    def _leds_off(self):
        if self.app.ctrl.connected:
            self.app.ctrl.lights_off()

    def _close(self):
        self._leds_off()
        self.destroy()

    def _open(self):
        name = filedialog.askopenfilename(
            parent=self, filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff")]
        )
        if name:
            try:
                key = self._key()
                self.images[key] = (np.asarray(Image.open(name).convert("RGB")), name)
                self.points[key] = []
                self.models.pop(self.role.get(), None)
                if self.role.get() == "point_light":
                    self._next_point()
                self._refresh()
            except Exception as exc:
                messagebox.showerror("Image error", str(exc), parent=self)

    def _capture(self):
        cam = self.app.cam
        if not cam.is_open:
            messagebox.showwarning("Camera required", "Open the camera first.", parent=self)

            return
        try:
            requested = float(self.exposure.get())
            if requested <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Exposure", "Exposure must be a positive number.", parent=self)

            return
        key, role = self._key(), self.role.get()
        self.status.set(f"Capturing at {requested:g} ms...")

        def shoot():
            actual = cam.set_exposure(requested)

            return actual, cam.capture_frame()

        def done(result, error):
            if error is not None:
                messagebox.showerror("Capture failed", str(error), parent=self)

                return
            actual, frame = result
            if frame is None:
                messagebox.showerror("Capture failed", "The camera returned no frame.", parent=self)
            elif abs(actual - requested) > EXPOSURE_TOLERANCE * requested:
                messagebox.showerror(
                    "Exposure not reachable",
                    f"The camera uses {actual:g} ms instead of {requested:g} ms; choose another exposure.",
                    parent=self,
                )
            else:
                self.images[key] = (
                    np.asarray(frame_to_pil(frame)),
                    f"camera frame @ {actual:g} ms",
                )
                self.points[key] = []
                self.models.pop(role, None)
                if role == "point_light":
                    self._next_point()
            self._refresh()

        self.app.run_async(shoot, done)

    def _refresh(self):
        self.canvas.delete("all")
        role = self.role.get()
        key = self._key()
        if key not in self.images:
            if role == "point_light":
                done = [str(i + 1) for i, k in enumerate(self._point_keys()) if k in self.images]
                self.status.set(
                    f"{self._label()}: press its LED button, then open or capture its chart image. "
                    f"Done: LEDs {', '.join(done) or 'none'}."
                )
            else:
                self.status.set(f"{role}: open or capture a 6 column x 5 row chart image.")

            return
        array, name = self.images[key]
        height, width = array.shape[:2]
        cw, ch = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        scale = min(cw / width, ch / height)
        self.scale = scale
        self.offset = ((cw - width * scale) / 2, (ch - height * scale) / 2)
        preview = Image.fromarray(array).resize((max(1, round(width * scale)), max(1, round(height * scale))))
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(*self.offset, image=self.photo, anchor="nw")
        pts = self.points.get(key, [])
        for index, (x, y) in enumerate(pts):
            px, py = self.offset[0] + x * scale, self.offset[1] + y * scale
            self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#ffce55", outline="black")
            self.canvas.create_text(px + 12, py - 12, text=str(index + 1), fill="white", anchor="w")
        if len(pts) == 4:
            coords = [v for x, y in pts for v in (self.offset[0] + x * scale, self.offset[1] + y * scale)]
            self.canvas.create_polygon(*coords, fill="", outline="#ffce55", width=2)
        progress = ""
        if role == "point_light":
            done = [
                str(i + 1)
                for i, k in enumerate(self._point_keys())
                if len(self.points.get(k, [])) == 4 and k in self.images
            ]
            progress = f" Corners set for LEDs {', '.join(done) or 'none'} of {len(self.led_bits)}."
        self.status.set(
            f"{self._label()}: {Path(name).name} · {len(pts)}/4 outer corners selected. "
            f"Order: top left -> top right -> bottom right -> bottom left.{progress}"
        )

    def _click(self, event):
        key = self._key()
        if key not in self.images or len(self.points[key]) >= 4:
            return
        array = self.images[key][0]
        x, y = (event.x - self.offset[0]) / self.scale, (event.y - self.offset[1]) / self.scale
        if 0 <= x < array.shape[1] and 0 <= y < array.shape[0]:
            self.points[key].append((x, y))
            self.models.pop(self.role.get(), None)
            self._refresh()

    def _reset(self):
        key = self._key()
        if key in self.points:
            self.points[key] = []
        self.models.pop(self.role.get(), None)
        self._refresh()

    def _patches(self, key):
        image = self.images[key][0]
        corners = np.asarray(self.points[key], dtype=float)
        if corners.shape != (4, 2):
            raise ValueError("Select all four outer chart corners first.")
        a, b = corners[1] - corners[0], corners[3] - corners[0]
        if abs(a[0] * b[1] - a[1] * b[0]) < 100:
            raise ValueError("Chart corners are too close together.")
        result = []
        for row in range(5):
            for col in range(6):
                u = (col + np.linspace(0.3, 0.7, 9)) / 6
                v = (row + np.linspace(0.3, 0.7, 9)) / 5
                uu, vv = np.meshgrid(u, v)
                xy = (
                    (1 - uu)[..., None] * (1 - vv)[..., None] * corners[0]
                    + uu[..., None] * (1 - vv)[..., None] * corners[1]
                    + uu[..., None] * vv[..., None] * corners[2]
                    + (1 - uu)[..., None] * vv[..., None] * corners[3]
                )
                xx = np.rint(xy[..., 0]).astype(int)
                yy = np.rint(xy[..., 1]).astype(int)
                if (
                    np.any(xx < 0)
                    or np.any(yy < 0)
                    or np.any(xx >= image.shape[1])
                    or np.any(yy >= image.shape[0])
                ):
                    raise ValueError("Chart patch falls outside the image. Check the corners.")
                result.append(np.median(image[yy, xx].reshape(-1, 3), axis=0))

        return np.asarray(result)

    def _fit(self):
        role = self.role.get()
        try:
            exposure = self.exposure.get()
            if exposure <= 0:
                raise ValueError("Exposure must be positive.")
            keys = ["diffuse"] if role == "diffuse" else self._point_keys()
            missing = [str(i + 1) for i, k in enumerate(keys) if k not in self.images]
            if missing:
                raise ValueError(f"Point light LED {', '.join(missing)} has no chart image yet.")
            patches = np.vstack([self._patches(k) for k in keys])
            # file names only: the bundle is copied into every run folder
            sources = basenames("; ".join(self.images[k][1] for k in keys))
            model = fit_model(patches, exposure_ms=exposure, source_file=sources)
            self.models[role] = model
            self.status.set(
                f"{role}: fitted {len(keys)} chart image(s), {len(patches)} patches, at {exposure:g} ms. "
                "Preview the result or select the other image type."
            )
        except (KeyError, ValueError) as exc:
            messagebox.showerror("Calibration error", str(exc), parent=self)

    def _preview(self):
        role = self.role.get()
        key = self._key()
        if role not in self.models:
            self._fit()
        if role not in self.models or key not in self.images:
            return
        try:
            source = Image.fromarray(self.images[key][0])
            source.thumbnail((700, 850))
            array = np.asarray(source)
            corrected = (
                apply_model(array, self.models[role], exposure_ms=self.models[role].exposure_ms) * 255
            ).astype("uint8")
            window = tk.Toplevel(self)
            window.title(f"{self._label()} · before / after")
            image = Image.fromarray(np.concatenate((array, corrected), axis=1))
            image.thumbnail((1400, 850))
            photo = ImageTk.PhotoImage(image)
            label = ttk.Label(window, image=photo)
            label.image = photo
            label.pack()
            ttk.Label(window, text="Original (left)    ·    Calibrated (right)").pack()
        except ValueError as exc:
            messagebox.showerror("Preview error", str(exc), parent=self)

    def _save(self):
        if set(self.models) != {"diffuse", "point_light"}:
            messagebox.showwarning(
                "Both images required",
                "Fit a diffuse and a point light chart image before saving.",
                parent=self,
            )

            return
        name = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".json",
            initialfile="color_calibration.json",
            filetypes=[("JSON", "*.json")],
        )
        if name:
            save_bundle(
                Path(name),
                ColorCalibration(1, self.models["diffuse"], self.models["point_light"]),
            )
            if hasattr(self.app, "load_color_calibration"):
                self.app.load_color_calibration(Path(name))
            self.status.set(f"Saved calibration to {name}. Measurements now correct images on save.")
