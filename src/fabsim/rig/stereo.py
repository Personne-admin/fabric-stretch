from __future__ import annotations

import argparse
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image

from . import ps_process
from .ps_process import PsOptions, RunData
from .settings import EXPOSURE_WARN_MS
from .ui.widgets import ImageCanvas, ThreadBridge


class PhotometricStereoTab:
    def __init__(self, parent, app):
        self.app = app
        self.root = parent

        self.run: RunData | None = None
        self._gains = None  # LED gains estimated for the loaded run
        self._cancel_event: threading.Event | None = None
        self._progress_visible = False

        self.run_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value="No run folder loaded. Pick a folder containing frames.json."
        )
        self.diffuse_prior_var = tk.BooleanVar(value=False)
        self.polarization_set_var = tk.StringVar(value="B")
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.ambient_var = tk.BooleanVar(value=False)
        self.estimate_gains_var = tk.BooleanVar(value=True)
        self.bilateral_var = tk.BooleanVar(value=False)
        self.frame_idx_var = tk.IntVar(value=0)

        self.video_albedo_var = tk.BooleanVar(value=True)
        self.video_normals_var = tk.BooleanVar(value=True)
        self.video_orig_var = tk.BooleanVar(value=False)
        self.video_skip_relax_var = tk.BooleanVar(value=True)
        self.fps_var = tk.IntVar(value=10)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar()
        self._build_ui()

    def _log(self, msg: str) -> None:
        self.app.post(self.app.append_log, msg)

    def _options(self) -> PsOptions:
        return PsOptions(
            ps_set=self.polarization_set_var.get(),
            gamma=float(self.gamma_var.get()),
            ambient_subtract=bool(self.ambient_var.get()),
            use_diffuse_albedo=bool(self.diffuse_prior_var.get()),
            bilateral_normals=bool(self.bilateral_var.get()),
            estimate_gains=bool(self.estimate_gains_var.get()),
            roi=self.canvas_in.roi,
        )

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=6)
        ttk.Button(top, text="Pick Run Folder", command=self._pick_run_folder).pack(
            side="left"
        )
        ttk.Entry(top, textvariable=self.run_dir_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=6
        )

        ttk.Label(
            self.root, textvariable=self.status_var, font=("", 10), foreground="#444"
        ).pack(fill="x", padx=10, pady=(0, 4))
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=2)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=290)
        left.pack(side="left", fill="y", padx=(10, 4), pady=6)
        left.pack_propagate(False)

        wf = ttk.LabelFrame(left, text="Solver options", padding=6)
        wf.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(
            wf, text="Use diffuse as albedo", variable=self.diffuse_prior_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            wf, text="Estimate LED gains", variable=self.estimate_gains_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            wf, text="Smooth normals (bilateral)", variable=self.bilateral_var
        ).pack(anchor="w")
        pol_row = ttk.Frame(wf)
        pol_row.pack(fill="x", pady=(4, 0))
        ttk.Label(pol_row, text="Polarization set:").pack(side="left")
        self.cmb_pol = ttk.Combobox(
            pol_row,
            textvariable=self.polarization_set_var,
            values=["A", "B"],
            state="disabled",
            width=4,
        )
        self.cmb_pol.pack(side="left", padx=4)

        cf = ttk.LabelFrame(left, text="Pre-processing", padding=6)
        cf.pack(fill="x", pady=(0, 4))
        r1 = ttk.Frame(cf)
        r1.pack(fill="x")
        ttk.Label(r1, text="Gamma:").pack(side="left")
        ttk.Entry(r1, textvariable=self.gamma_var, width=8).pack(side="left", padx=4)
        ttk.Checkbutton(cf, text="Ambient subtract", variable=self.ambient_var).pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Button(
            cf, text="Clear ROI", command=lambda: self.canvas_in.clear_roi()
        ).pack(anchor="w", pady=(4, 0))

        bf = ttk.LabelFrame(left, text="Batch / video", padding=6)
        bf.pack(fill="x", pady=(0, 4))
        ttk.Button(bf, text="Process All Frames", command=self._process_all).pack(
            fill="x", pady=2
        )
        ttk.Button(bf, text="Make Video", command=self._make_video).pack(
            fill="x", pady=2
        )
        ttk.Separator(bf, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(bf, text="Video tracks:").pack(anchor="w")
        ttk.Checkbutton(bf, text="Albedo", variable=self.video_albedo_var).pack(
            anchor="w"
        )
        ttk.Checkbutton(bf, text="Normals", variable=self.video_normals_var).pack(
            anchor="w"
        )
        ttk.Checkbutton(
            bf, text="Original (diffuse/rgb)", variable=self.video_orig_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            bf, text="Skip relaxation holds", variable=self.video_skip_relax_var
        ).pack(anchor="w")
        fr = ttk.Frame(bf)
        fr.pack(fill="x", pady=(4, 0))
        ttk.Label(fr, text="FPS:").pack(side="left")
        ttk.Spinbox(fr, from_=1, to=60, textvariable=self.fps_var, width=5).pack(
            side="left", padx=4
        )

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=6)

        fs = ttk.Frame(right)
        fs.pack(fill="x")
        ttk.Label(fs, text="Frame:").pack(side="left")
        self.frame_combo = ttk.Combobox(
            fs, textvariable=self.frame_idx_var, width=8, state="readonly", values=[]
        )
        self.frame_combo.pack(side="left", padx=4)
        self.frame_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_preview())
        ttk.Button(fs, text="Preview", command=self._preview_current).pack(
            side="left", padx=8
        )
        self.lbl_frame_meta = ttk.Label(fs, text="")
        self.lbl_frame_meta.pack(side="left", padx=4)

        panes = ttk.Frame(right)
        panes.pack(fill="both", expand=True, pady=(6, 0))
        in_box = ttk.LabelFrame(panes, text="Input (drag ROI on white area)")
        in_box.pack(side="left", fill="both", expand=True, padx=(0, 2))
        self.canvas_in = ImageCanvas(in_box, allow_roi=True, bg="#202020")
        self.canvas_in.pack(fill="both", expand=True)

        out_nb = ttk.Notebook(panes)
        out_nb.pack(side="left", fill="both", expand=True, padx=(2, 0))
        tab_alb, tab_nrm = ttk.Frame(out_nb), ttk.Frame(out_nb)
        out_nb.add(tab_alb, text="Albedo")
        out_nb.add(tab_nrm, text="Normals")
        self.canvas_alb = ImageCanvas(tab_alb, bg="#101010")
        self.canvas_alb.pack(fill="both", expand=True)
        self.canvas_nrm = ImageCanvas(tab_nrm, bg="#101010")
        self.canvas_nrm.pack(fill="both", expand=True)

        self._prog_frame = ttk.Frame(self.root)
        ttk.Label(self._prog_frame, textvariable=self.progress_text_var).pack(
            side="left", padx=4
        )
        ttk.Progressbar(
            self._prog_frame, variable=self.progress_var, maximum=100.0, length=260
        ).pack(side="left", padx=4)
        self._prog_cancel = ttk.Button(
            self._prog_frame, text="Cancel", command=self._do_cancel
        )
        self._prog_cancel.pack(side="left", padx=4)

    def _pick_run_folder(self):
        initial = (
            str(self.app.project_dir)
            if getattr(self.app, "project_dir", None)
            else str(Path.home())
        )
        folder = filedialog.askdirectory(
            initialdir=initial, title="Pick a run folder containing frames.json"
        )
        if folder:
            self._load_run_folder(Path(folder))

    def _load_run_folder(self, folder: Path):
        try:
            run = RunData(folder)
        except FileNotFoundError:
            messagebox.showerror(
                "Not a run folder", f"frames.json not found in:\n{folder}"
            )

            return
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot load run", f"{folder}\n\n{exc}")

            return

        self.run = run
        self._gains = None
        self.run_dir_var.set(str(folder))
        if run.color_bundle is not None:
            self._log(
                f"[PS] colour calibration {run.color_path} (already applied at capture: "
                f"{sorted(run.already_corrected) or 'none'})"
            )
        exposure = run.exposure_ms
        if exposure is not None and exposure >= EXPOSURE_WARN_MS:
            self._log(
                f"[PS] WARN: acquisition exposure is {EXPOSURE_WARN_MS:g} ms or higher; "
                "inspect images for clipping"
            )
        self._log(
            f"[PS] light: {run.light.source}, azimuths {run.light.azimuths_deg}, "
            f"elevation {run.light.elevation_deg}"
        )
        self.cmb_pol.config(
            state="readonly" if run.caps.get("has_polarization") else "disabled"
        )
        n = run.caps.get("frame_count", 0)
        self.frame_combo["values"] = [str(i) for i in range(n)]
        if n > 0:
            self.frame_combo.current(0)
            self.frame_idx_var.set(0)
            self._load_preview()
        else:
            for canvas in (self.canvas_in, self.canvas_alb, self.canvas_nrm):
                canvas.clear()
        self._refresh_status()
        self._log(f"[PS] Loaded run: {folder.name} ({n} frames, mode={run.mode})")

    def _refresh_status(self):
        if self.run is None:
            self.status_var.set(
                "No run folder loaded. Pick a folder containing frames.json."
            )

            return
        caps = self.run.caps
        parts = [f"{caps.get('frame_count', 0)} frames", f"mode: {self.run.mode}"]
        if caps.get("has_diffuse"):
            parts.append("diffuse")
        if caps.get("ps_count"):
            parts.append(f"{caps['ps_count']} PS shots")
        if caps.get("has_polarization"):
            parts.append("polarization")
        if self.run.mode != "white":
            parts.append("no PS images: preview only")
        parts.append(f"light: {self.run.light.source}")
        self.status_var.set(" | ".join(parts))

    def _frame_index(self) -> int:
        try:
            return int(self.frame_idx_var.get())
        except (ValueError, tk.TclError):
            return 0

    def _input_preview(self, idx: int) -> tuple[np.ndarray | None, str]:
        run = self.run
        keys = ["image_diffuse", "image_rgb"] + sorted(
            k for k in run.frames[idx] if k.startswith("image_")
        )
        for key in keys:
            image = run.read_bgr(idx, key)
            if image is not None:
                return image, key

        return None, "(none)"

    def _load_preview(self):
        if self.run is None:
            return
        idx = self._frame_index()
        if not 0 <= idx < self.run.caps.get("frame_count", 0):
            return
        image, label = self._input_preview(idx)
        if image is None:
            self.canvas_in.clear()
            self.lbl_frame_meta.config(text="No image")

            return
        self.canvas_in.show_image(Image.fromarray(image[:, :, ::-1]))
        frame = self.run.frames[idx]
        relax = " (relax)" if frame.get("is_relaxation_hold") else ""
        self.lbl_frame_meta.config(
            text=f"step={frame.get('step_index')} disp={frame.get('displacement_mm', 0):.2f}mm"
            f"{relax} [{label}]"
        )
        self.canvas_alb.clear()
        self.canvas_nrm.clear()

    def _preview_current(self):
        if self.run is None:
            messagebox.showinfo("PS", "Load a run folder first.")

            return
        run, idx, options = self.run, self._frame_index(), self._options()

        def work():
            result = ps_process.process_frame(run, idx, options, self._gains)

            return (
                ps_process.albedo_preview_u8(result.albedo_rgb),
                ps_process.normals_preview_u8(
                    result.normals, options.bilateral_normals
                ),
                result.gains,
            )

        def done(result, error):
            if error is not None:
                messagebox.showerror("PS error", str(error))

                return
            albedo, normals, gains = result
            self.canvas_alb.show_image(Image.fromarray(albedo))
            self.canvas_nrm.show_image(Image.fromarray(normals))
            self._log(f"[PS] frame {idx}: LED gains {np.round(gains, 3).tolist()}")

        self.app.run_async(work, done)

    def _show_progress(self, text):
        if not self._progress_visible:
            self._prog_frame.pack(fill="x", padx=10, pady=(0, 4))
            self._progress_visible = True
        self.progress_var.set(0)
        self.progress_text_var.set(text)
        self._prog_cancel.config(state="normal")

    def _update_progress(self, pct, text=None):
        self.progress_var.set(max(0, min(100, pct)))
        if text:
            self.progress_text_var.set(text)

    def _hide_progress(self):
        if self._progress_visible:
            self._prog_frame.pack_forget()
            self._progress_visible = False

    def _do_cancel(self):
        if self._cancel_event:
            self._cancel_event.set()
            self._prog_cancel.config(state="disabled")
            self.progress_text_var.set("Cancelling")

    def _out_dir(self) -> Path:
        return self.run.run_dir / "ps_output"

    def _process_all(self):
        if self.run is None:
            messagebox.showinfo("PS", "Load a run folder first.")

            return
        if self._cancel_event is not None:
            return
        run, options = self.run, self._options()
        out_dir = self._out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        total = len(run.frames)
        cancel = self._cancel_event = threading.Event()
        self._show_progress(f"Batch: 0/{total}")
        post = self.app.post

        def worker():
            gains = self._gains
            results, errors = [], 0
            for i, frame in enumerate(run.frames):
                if cancel.is_set():
                    break
                try:
                    result = ps_process.process_frame(run, i, options, gains)
                    gains = result.gains if gains is None else gains
                    fid = frame.get("frame_id", i)
                    alb_path = out_dir / f"img_{fid:04d}_albedo.png"
                    nrm_path = out_dir / f"img_{fid:04d}_normals.png"
                    cv2.imwrite(
                        str(alb_path),
                        ps_process.albedo_preview_u8(result.albedo_rgb)[:, :, ::-1],
                    )
                    cv2.imwrite(
                        str(nrm_path),
                        ps_process.normals_preview_u8(
                            result.normals, options.bilateral_normals
                        )[:, :, ::-1],
                    )
                    results.append(
                        {
                            "frame_id": fid,
                            "step_index": frame.get("step_index"),
                            "displacement_mm": frame.get("displacement_mm"),
                            "is_relaxation_hold": bool(frame.get("is_relaxation_hold")),
                            "albedo": alb_path.name,
                            "normals": nrm_path.name,
                        }
                    )
                except Exception as exc:
                    errors += 1
                    self._log(f"[PS ERROR] frame {i}: {exc}")
                post(
                    self._update_progress,
                    (i + 1) / total * 100,
                    f"Batch: {i + 1}/{total}",
                )

            try:
                data = ps_process.manifest(
                    run, options, gains if gains is not None else np.ones(3), results
                )
                (out_dir / "ps_manifest.json").write_text(
                    json.dumps(data, indent=2) + "\n", encoding="utf-8"
                )
            except Exception as exc:
                self._log(f"[PS] manifest write failed: {exc}")
            post(self._batch_done, len(results), errors, out_dir, gains)

        threading.Thread(target=worker, daemon=True).start()

    def _batch_done(self, processed: int, errors: int, out_dir: Path, gains):
        self._gains = gains
        self._hide_progress()
        self._cancel_event = None
        messagebox.showinfo(
            "PS", f"Batch done.\nProcessed: {processed}\nErrors: {errors}\n{out_dir}"
        )

    def _make_video(self):
        if self.run is None:
            messagebox.showinfo("PS", "Load a run folder first.")

            return
        if self._cancel_event is not None:
            return
        want_alb, want_nrm, want_org = (
            self.video_albedo_var.get(),
            self.video_normals_var.get(),
            self.video_orig_var.get(),
        )
        if not (want_alb or want_nrm or want_org):
            messagebox.showinfo("PS", "Select at least one video track.")

            return

        run = self.run
        fps = max(1, min(60, self.fps_var.get()))
        out_dir = self._out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = run.frames
        if self.video_skip_relax_var.get():
            frames = [f for f in frames if not f.get("is_relaxation_hold")]

        sets: dict[str, list[Path]] = {}
        if want_alb:
            sets["albedo"] = [
                out_dir / f"img_{f.get('frame_id', i):04d}_albedo.png"
                for i, f in enumerate(frames)
            ]
        if want_nrm:
            sets["normals"] = [
                out_dir / f"img_{f.get('frame_id', i):04d}_normals.png"
                for i, f in enumerate(frames)
            ]
        if want_org:
            keys = run.caps.get("all_image_keys", [])
            org_key = next(
                (k for k in ("image_diffuse", "image_rgb") if k in keys),
                keys[0] if keys else None,
            )
            if org_key:
                sets["original"] = [
                    run.run_dir / "images" / f[org_key]
                    for f in frames
                    if f.get(org_key)
                ]
        sets = {
            label: [p for p in paths if p.exists()] for label, paths in sets.items()
        }
        sets = {label: paths for label, paths in sets.items() if paths}

        total = sum(len(v) for v in sets.values())
        if total == 0:
            messagebox.showerror(
                "PS", "No frames available. Run 'Process All' first for albedo/normals."
            )

            return

        cancel = self._cancel_event = threading.Event()
        self._show_progress(f"Video: 0/{total}")
        post = self.app.post

        def worker():
            written, results, cancelled = 0, [], False
            for label, paths in sets.items():
                if cancel.is_set():
                    cancelled = True
                    break
                out_path = out_dir / f"ps_{label}.mp4"
                first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
                if first is None:
                    continue
                h, w = first.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (w, h),
                    True,
                )
                if not writer or not writer.isOpened():
                    out_path = out_path.with_suffix(".avi")
                    writer = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*"XVID"),
                        float(fps),
                        (w, h),
                        True,
                    )
                if not writer or not writer.isOpened():
                    self._log(f"[PS] cannot open writer for {label}")
                    continue
                for p in paths:
                    if cancel.is_set():
                        cancelled = True
                        break
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h))
                    writer.write(img)
                    written += 1
                    post(
                        self._update_progress,
                        written / total * 100,
                        f"Video ({label}): {written}/{total}",
                    )
                writer.release()
                if cancelled:
                    out_path.unlink(missing_ok=True)
                else:
                    results.append(str(out_path))
                    self._log(f"[PS] {label}: {out_path}")
            post(self._video_done, cancelled, results)

        threading.Thread(target=worker, daemon=True).start()

    def _video_done(self, cancelled: bool, results: list[str]):
        self._hide_progress()
        self._cancel_event = None
        if cancelled:
            messagebox.showinfo("PS", "Video creation cancelled.")
        elif results:
            messagebox.showinfo("PS", "Videos:\n" + "\n".join(results))
        else:
            messagebox.showerror("PS", "No videos created.")


class _StandaloneApp(ThreadBridge, tk.Tk):

    project_dir: Path | None = None

    def __init__(self, run_folder: Path | None = None):
        super().__init__()
        self.title("Fabric Stretch - Photometric Stereo")
        self.geometry("1100x760")
        self._init_bridge()
        self.tab = PhotometricStereoTab(self, self)
        if run_folder is not None:
            self.after_idle(self.tab._load_run_folder, run_folder)

    def append_log(self, text: str) -> None:
        print(text, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fabsim stereo", description="Photometric stereo GUI for rig runs"
    )
    parser.add_argument(
        "run", nargs="?", type=Path, help="run folder (with frames.json) to load"
    )
    args = parser.parse_args(argv)
    _StandaloneApp(args.run).mainloop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
