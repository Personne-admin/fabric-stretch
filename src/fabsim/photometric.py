from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "LightConfig",
    "albedo_scale",
    "color_albedo",
    "decode_albedo_u16",
    "decode_normals_u16",
    "default_led_config_path",
    "encode_albedo_u8",
    "encode_albedo_u16",
    "encode_normals_u16",
    "estimate_gains",
    "albedo_correlation",
    "exposure_gain",
    "light_direction",
    "light_matrix",
    "load_light_config",
    "read_ps_gray",
    "read_rgb_linear",
    "remove_normal_tilt",
    "render",
    "shade",
    "mean_normal_tilt_deg",
    "solve",
    "support_mask",
]

#: Rec. 601 luma weights (R, G, B), the same as ``cv2.cvtColor(..., COLOR_BGR2GRAY)``.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

_EPS = 1e-8
#: Albedo versus diffuse-luma correlation below which the photometric layers are flagged as unreliable
#: (good captures reach 0.86 to 0.95; a dim or mis-lit capture drops far below).
MIN_ALBEDO_CORRELATION = 0.6


@dataclass(frozen=True)
class LightConfig:
    azimuths_deg: tuple[float, float, float]  # LED order = image_ps_1..3
    elevation_deg: float
    gains: tuple[float, float, float] | None = None  # radiance relative to the mean; None = unknown
    source: str = ""  # provenance string written to manifests

    def with_gains(self, gains) -> LightConfig:
        return dataclasses.replace(self, gains=tuple(float(g) for g in gains))

    def to_dict(self) -> dict:
        out = {
            "azimuths_deg": [float(a) for a in self.azimuths_deg],
            "elevation_deg": float(self.elevation_deg),
            "source": self.source,
        }
        if self.gains is not None:
            out["gains"] = [float(g) for g in self.gains]

        return out


def default_led_config_path() -> Path:
    return Path(str(resources.files("fabsim.rig") / "defaults" / "led_config.json"))


def _from_dict(data: dict, source: str) -> LightConfig:
    azimuths = data.get("light_azimuths_deg", data.get("azimuths_deg"))
    elevation = data.get("light_elevation_deg", data.get("elevation_deg"))
    if azimuths is None or elevation is None or len(azimuths) != 3:
        raise ValueError(f"LED configuration ({source}) needs three azimuths and an elevation")

    gains = data.get("gains")

    return LightConfig(
        azimuths_deg=tuple(float(a) for a in azimuths),
        elevation_deg=float(elevation),
        gains=None if gains is None else tuple(float(g) for g in gains),
        source=source,
    )


def load_light_config(path: str | Path | None = None, session: dict | None = None) -> LightConfig:
    if session and isinstance(session.get("led_config"), dict):
        return _from_dict(session["led_config"], "session.json:led_config")

    if path is not None:
        path = Path(path)

        return _from_dict(json.loads(path.read_text()), f"file:{path.name}")

    default = default_led_config_path()

    return _from_dict(json.loads(default.read_text()), "packaged:led_config.json")


def light_direction(azimuth_deg, elevation_deg: float) -> np.ndarray:
    e = np.deg2rad(elevation_deg)
    a = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))

    return np.stack(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.full_like(a, np.sin(e))],
        axis=-1,
    )


def light_matrix(cfg: LightConfig) -> np.ndarray:
    return light_direction(cfg.azimuths_deg, cfg.elevation_deg)


def _gains_or_ones(cfg: LightConfig, gains) -> np.ndarray:
    if gains is None:
        gains = cfg.gains

    if gains is None:
        return np.ones(3, dtype=np.float32)

    gains = np.asarray(gains, dtype=np.float32)
    if gains.shape != (3,) or np.any(gains <= 0):
        raise ValueError("gains must be three positive numbers")

    return gains


def support_mask(ps_gray: np.ndarray, rel: float = 0.05) -> np.ndarray:
    mean = np.asarray(ps_gray, dtype=np.float32).mean(axis=-1)

    return mean > rel * max(float(np.median(mean)), 1e-3)


def estimate_gains(ps_gray: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
    ps = np.asarray(ps_gray, dtype=np.float32)
    pixels = ps[support] if support is not None else ps.reshape(-1, 3)
    if pixels.shape[0] == 0:
        return np.ones(3)

    medians = np.median(pixels, axis=0).astype(np.float64)
    if not np.all(medians > 0):
        return np.ones(3)

    return medians / medians.mean()


def solve(ps_gray: np.ndarray, cfg: LightConfig, gains=None) -> tuple[np.ndarray, np.ndarray]:
    ps = np.asarray(ps_gray, dtype=np.float32)
    if ps.ndim != 3 or ps.shape[-1] != 3:
        raise ValueError(f"ps_gray must be (H, W, 3), got {ps.shape}")

    pinv = np.linalg.pinv(light_matrix(cfg)).astype(np.float32)
    g = (ps / _gains_or_ones(cfg, gains)) @ pinv.T

    albedo = np.linalg.norm(g, axis=-1)
    normals = g / np.maximum(albedo, _EPS)[..., None]
    normals[albedo < _EPS] = (0.0, 0.0, 1.0)

    return albedo, normals


def render(albedo_gray: np.ndarray, normals: np.ndarray, cfg: LightConfig, gains=None) -> np.ndarray:
    light = light_matrix(cfg).astype(np.float32)
    shading = np.maximum(np.asarray(normals, dtype=np.float32) @ light.T, 0.0)

    return np.asarray(albedo_gray, dtype=np.float32)[..., None] * shading * _gains_or_ones(cfg, gains)


def shade(
    albedo: np.ndarray,
    normals: np.ndarray,
    direction,
    intensity: float = 1.0,
    ambient: float = 0.0,
    gain: float = 1.0,
) -> np.ndarray:
    diffuse = np.maximum(
        np.asarray(normals, dtype=np.float32) @ np.asarray(direction, dtype=np.float32),
        0.0,
    )

    factor = ambient + gain * intensity * diffuse
    a = np.asarray(albedo, dtype=np.float32)
    if a.ndim == 3:
        factor = factor[..., None]

    return np.clip(a * factor, 0.0, 1.0)


def color_albedo(albedo_gray: np.ndarray, diffuse_rgb_linear: np.ndarray) -> np.ndarray:
    rgb = np.asarray(diffuse_rgb_linear, dtype=np.float32)
    luma = np.maximum(rgb @ LUMA, 1e-6)

    return np.asarray(albedo_gray, dtype=np.float32)[..., None] * (rgb / luma[..., None])


def albedo_scale(albedo_rgb: np.ndarray, support: np.ndarray | None = None) -> float:
    a = np.asarray(albedo_rgb, dtype=np.float32)
    peak = a.max(axis=-1) if a.ndim == 3 else a

    values = peak[support] if support is not None else peak.reshape(-1)
    if values.size == 0:
        return 1.0

    scale = float(np.percentile(values, 99.9))

    return scale if np.isfinite(scale) and scale > _EPS else 1.0


def encode_normals_u16(n: np.ndarray) -> np.ndarray:
    v = (np.clip(np.asarray(n, dtype=np.float32), -1.0, 1.0) * 0.5 + 0.5) * 65535.0

    return np.rint(v).astype(np.uint16)


def decode_normals_u16(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float32) * (2.0 / 65535.0) - 1.0


def encode_albedo_u16(a: np.ndarray, scale: float) -> np.ndarray:
    return np.rint(np.clip(np.asarray(a, dtype=np.float32) / scale, 0.0, 1.0) * 65535.0).astype(np.uint16)


def decode_albedo_u16(v: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(v, dtype=np.float32) * (scale / 65535.0)


def encode_albedo_u8(a: np.ndarray, scale: float) -> np.ndarray:
    return np.rint(np.clip(np.asarray(a, dtype=np.float32) / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def read_ps_gray(paths) -> np.ndarray:
    channels = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot read image {path}")

        channels.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)

    if len({c.shape for c in channels}) != 1:
        raise ValueError("PS images differ in size")

    return np.stack(channels, axis=-1)


def read_rgb_linear(path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read image {path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def mean_normal_tilt_deg(normals: np.ndarray, support: np.ndarray | None = None) -> float:
    n = np.asarray(normals, dtype=np.float32)

    mean = (n[support] if support is not None else n.reshape(-1, 3)).mean(axis=0, dtype=np.float64)

    length = float(np.linalg.norm(mean))

    return 0.0 if length < _EPS else float(np.degrees(np.arccos(np.clip(mean[2] / length, -1.0, 1.0))))


def remove_normal_tilt(normals: np.ndarray, max_samples: int = 200_000) -> tuple[np.ndarray, float]:
    n = np.asarray(normals, dtype=np.float32)
    h, w = n.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_samples))))
    u = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    uu, vv = np.meshgrid(u[::step], v[::step])
    design = np.stack([np.ones(uu.size, np.float32), uu.reshape(-1), vv.reshape(-1)], axis=1)

    sample = n[::step, ::step, :2].reshape(-1, 2)
    coefficients = np.linalg.lstsq(design, sample, rcond=None)[0]  # (3, 2)
    plane = coefficients[0] + u[None, :, None] * coefficients[1] + v[:, None, None] * coefficients[2]

    out = n.copy()
    out[..., :2] -= plane
    length = np.linalg.norm(out, axis=-1, keepdims=True)
    out = np.where(length > _EPS, out / np.maximum(length, _EPS), np.float32([0.0, 0.0, 1.0]))

    return out.astype(np.float32), mean_normal_tilt_deg(n)


def _luma(rgb_linear: np.ndarray) -> np.ndarray:
    return np.asarray(rgb_linear, dtype=np.float32) @ LUMA


def exposure_gain(
    albedo_gray: np.ndarray,
    normals: np.ndarray,
    diffuse_rgb_linear: np.ndarray,
    support: np.ndarray | None = None,
) -> float:
    head_on = np.asarray(albedo_gray, dtype=np.float32) * np.maximum(np.asarray(normals)[..., 2], 0.0)
    diffuse = _luma(diffuse_rgb_linear)
    if support is not None:
        head_on, diffuse = head_on[support], diffuse[support]

    denominator = float(head_on.mean()) if head_on.size else 0.0
    value = float(diffuse.mean()) / denominator if denominator > _EPS else 1.0

    return value if np.isfinite(value) and value > _EPS else 1.0


def albedo_correlation(
    albedo_gray: np.ndarray,
    diffuse_rgb_linear: np.ndarray,
    support: np.ndarray | None = None,
) -> float:
    a = np.asarray(albedo_gray, dtype=np.float32)
    d = _luma(diffuse_rgb_linear)
    if support is not None:
        a, d = a[support], d[support]

    a, d = a.reshape(-1), d.reshape(-1)
    if a.size < 2 or float(a.std()) < _EPS or float(d.std()) < _EPS:
        return 0.0

    return float(np.corrcoef(a, d)[0, 1])
