from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import photometric
from ..color_calibration import ColorCalibration, apply_model, load_bundle
from . import config
from .colorcorrect import corrected_models

#: Albedo values are clipped to ``[0, ALBEDO_VISUALIZATION_MAX]`` in the 8-bit preview PNGs.
ALBEDO_VISUALIZATION_MAX = 1.0
NORMAL_FLAT = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def detect_mode_from_frames(frames_data: dict) -> tuple[str, dict]:
    frames = frames_data.get("frames", [])
    if not frames:
        return "unknown", {}
    keys = sorted(k for k in frames[0] if k.startswith("image_"))
    ps_a = sorted(k for k in keys if re.match(r"^image_ps_\d+$", k))
    ps_b = sorted(k for k in keys if re.match(r"^image_ps_b_\d+$", k))
    caps = {
        "has_rgb": "image_rgb" in keys,
        "has_diffuse": "image_diffuse" in keys,
        "has_polarization": bool(ps_b),
        "ps_a_keys": ps_a,
        "ps_b_keys": ps_b,
        "ps_count": len(ps_a),
        "frame_count": len(frames),
        "all_image_keys": keys,
    }
    if ps_a:
        return "white", caps
    if caps["has_rgb"]:
        return "rgb", caps

    return "unknown", caps


@dataclass(frozen=True)
class PsOptions:
    ps_set: str = "A"  # "A" or "B" (polarised set, if captured)
    gamma: float = 1.0  # 1.0 = camera bytes are linear (the rig's convention)
    ambient_subtract: bool = False
    use_diffuse_albedo: bool = False
    bilateral_normals: bool = False  # smooth the 8-bit normal preview only
    estimate_gains: bool = True  # estimate LED radiances unless the configuration has them
    roi: tuple[int, int, int, int] | None = None  # white reference region (x, y, w, h)


@dataclass
class PsResult:
    albedo_rgb: np.ndarray  # (H, W, 3) linear RGB, zero outside the support
    albedo_gray: np.ndarray
    normals: np.ndarray  # (H, W, 3) unit vectors, camera frame
    support: np.ndarray
    gains: np.ndarray


class RunData:
    def __init__(self, run_dir: Path, led_config_path: Path | None = None):
        self.run_dir = Path(run_dir)
        frames_path = self.run_dir / "frames.json"
        if not frames_path.is_file():
            raise FileNotFoundError(f"{frames_path} not found - not a run folder")
        self.frames_data = json.loads(frames_path.read_text(encoding="utf-8"))
        session_path = self.run_dir / "session.json"
        self.session = json.loads(session_path.read_text(encoding="utf-8")) if session_path.is_file() else {}
        self.mode, self.caps = detect_mode_from_frames(self.frames_data)
        self.light = resolve_light(self.session, led_config_path)
        camera = self.session.get("camera") or {}
        self.exposure_ms = camera.get("exposure_ms_actual", self.session.get("cam_exposure_ms"))
        self.color_stamp = self.session.get("color_calibration")
        self.already_corrected = corrected_models(self.color_stamp)
        self.color_path: Path | None = None
        self.color_bundle: ColorCalibration | None = None
        for candidate in (
            self.run_dir / "color_calibration.json",
            self.run_dir.parent / "color_calibration.json",
        ):
            if candidate.is_file():
                self.color_path, self.color_bundle = candidate, load_bundle(candidate)
                break

    @property
    def frames(self) -> list[dict]:
        return self.frames_data["frames"]

    def ps_keys(self, ps_set: str = "A") -> list[str]:
        if ps_set == "B" and self.caps.get("ps_b_keys"):
            return self.caps["ps_b_keys"]

        return self.caps.get("ps_a_keys", [])

    def image_path(self, idx: int, key: str) -> Path | None:
        name = self.frames[idx].get(key)

        return self.run_dir / "images" / name if name else None

    def read_bgr(self, idx: int, key: str) -> np.ndarray | None:
        path = self.image_path(idx, key)
        if path is None or not path.is_file():
            return None

        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    def calibrated(self, image_bgr: np.ndarray | None, model_name: str) -> np.ndarray | None:
        if image_bgr is None or self.color_bundle is None or model_name in self.already_corrected:
            return image_bgr
        if self.exposure_ms is None:
            raise RuntimeError("color calibration requires cam_exposure_ms in session.json")
        model = getattr(self.color_bundle, model_name)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        corrected = apply_model(rgb, model, exposure_ms=float(self.exposure_ms))

        return cv2.cvtColor((corrected * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def resolve_light(session: dict, led_config_path: Path | None = None) -> photometric.LightConfig:
    if led_config_path is None:
        user_file = config.config_dir() / config.LED_CONFIG_FILE
        led_config_path = user_file if user_file.is_file() else None

    return photometric.load_light_config(led_config_path, session)


def preprocess_gray(img_bgr: np.ndarray, options: PsOptions) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gamma = options.gamma if options.gamma and options.gamma > 0 else 1.0
    gray = np.clip(gray, 0, 1) ** gamma
    if options.roi is not None:
        x, y, w, h = options.roi
        patch = gray[max(0, y) : y + h, max(0, x) : x + w]
        if patch.size >= 100 and float(np.median(patch)) > 1e-6:
            gray = gray / float(np.median(patch))
    if options.ambient_subtract:
        gray = np.clip(gray - np.quantile(gray, 0.01), 0, None)

    return gray


def load_ps_images(run: RunData, idx: int, options: PsOptions) -> list[np.ndarray]:
    if run.mode != "white":
        raise RuntimeError(f"photometric stereo needs image_ps_1..3 images; this run is {run.mode!r}")
    keys = run.ps_keys(options.ps_set)
    if len(keys) != 3:
        raise RuntimeError(f"frame {idx}: need exactly 3 PS images, the run has {len(keys)}")
    images = []
    for key in keys:
        image = run.calibrated(run.read_bgr(idx, key), "point_light")
        if image is None:
            raise RuntimeError(f"frame {idx}: missing {key}")
        images.append(image)

    return images


def solve_gray(run: RunData, idx: int, options: PsOptions, gains=None):
    images = load_ps_images(run, idx, options)
    ps_gray = np.stack([preprocess_gray(img, options) for img in images], axis=-1)
    support = photometric.support_mask(ps_gray)
    if gains is None and run.light.gains is None and options.estimate_gains:
        gains = photometric.estimate_gains(ps_gray, support)
    albedo, normals = photometric.solve(ps_gray, run.light, gains)
    used = gains if gains is not None else run.light.gains
    used = np.ones(3) if used is None else np.asarray(used, dtype=np.float64)

    return ps_gray, albedo, normals, support, used, images


def process_frame(run: RunData, idx: int, options: PsOptions, gains=None) -> PsResult:
    _ps_gray, albedo, normals, support, used, images = solve_gray(run, idx, options, gains)
    diffuse = run.calibrated(run.read_bgr(idx, "image_diffuse"), "diffuse")
    if diffuse is not None:
        color = cv2.cvtColor(diffuse, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if options.use_diffuse_albedo:
            albedo = preprocess_gray(diffuse, options)
    else:
        color = np.mean(
            [cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for img in images],
            axis=0,
        )
    albedo_rgb = photometric.color_albedo(albedo, color)
    albedo_rgb[~support] = 0.0
    normals[~support] = NORMAL_FLAT

    return PsResult(albedo_rgb, albedo, normals, support, used)


def albedo_preview_u8(albedo_rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(albedo_rgb / ALBEDO_VISUALIZATION_MAX, 0, 1) * 255).astype(np.uint8)


def normals_preview_u8(normals: np.ndarray, bilateral: bool = False) -> np.ndarray:
    encoded = np.rint(np.clip(normals * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)

    return cv2.bilateralFilter(encoded, 5, 50, 50) if bilateral else encoded


def manifest(run: RunData, options: PsOptions, gains, frames: list[dict]) -> dict:
    return {
        "solver": "fabsim.photometric.solve",
        "light": run.light.to_dict(),
        "gains": [float(g) for g in gains],
        "ps_set": options.ps_set,
        "use_diffuse_albedo": options.use_diffuse_albedo,
        "gamma": options.gamma,
        "ambient_subtract": options.ambient_subtract,
        "bilateral_normals": options.bilateral_normals,
        "white_calib_roi": list(options.roi) if options.roi else None,
        "albedo_visualization_max": ALBEDO_VISUALIZATION_MAX,
        "color_calibration": (
            None
            if run.color_bundle is None
            else {
                "file": run.color_path.name,
                "bundle_version": run.color_bundle.version,
                "acquisition_exposure_ms": run.exposure_ms,
                "already_applied_at_capture": sorted(run.already_corrected),
            }
        ),
        "color_calibration_at_capture": run.color_stamp,
        "frames": frames,
    }
