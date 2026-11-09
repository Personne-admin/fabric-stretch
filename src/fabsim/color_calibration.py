from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

REFERENCE_LAB = np.array(
    [
        [39.12, 13.24, 15.07],
        [65.43, 18.11, 18.72],
        [49.87, -4.34, -22.29],
        [44.26, -13.80, 22.85],
        [55.56, 9.82, -24.49],
        [70.82, -33.43, -0.35],
        [50.87, -27.17, -29.46],
        [97.06, -0.40, 1.13],
        [92.02, -0.60, 0.23],
        [87.34, -0.75, 0.21],
        [82.14, -1.06, 0.43],
        [63.51, 34.26, 59.60],
        [52.79, 50.88, -12.72],
        [72.06, -1.19, 0.28],
        [62.15, -1.07, 0.19],
        [49.25, -0.16, 0.01],
        [38.62, -0.18, -0.04],
        [39.92, 11.81, -46.07],
        [82.74, 3.45, 81.29],
        [28.86, 0.54, 0.60],
        [16.19, -0.05, 0.73],
        [8.29, -0.81, 0.19],
        [3.44, -0.23, 0.49],
        [52.24, 48.55, 18.51],
        [43.96, 52.00, 30.01],
        [54.91, -38.91, 30.77],
        [29.37, 13.06, -49.49],
        [72.95, 16.83, 68.80],
        [72.46, -24.45, 55.93],
        [31.41, 20.98, -19.43],
    ],
    dtype=np.float64,
)
D50 = np.array([0.96422, 1.0, 0.82521])
D65 = np.array([0.95047, 1.0, 1.08883])
BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)
XYZ_TO_RGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.969266, 1.8760108, 0.041556],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


@dataclasses.dataclass(frozen=True)
class CalibrationModel:
    version: int
    fit_type: Literal["linear", "affine", "rootpoly2"]
    coefficients: list[list[float]]
    input_encoding: Literal["linear", "srgb"]
    input_scale: float
    reference_white: str
    output_space: str
    exposure_ms: float
    channel_order: str
    source_file: str
    fitted_at: str


@dataclasses.dataclass(frozen=True)
class ColorCalibration:
    version: int
    diffuse: CalibrationModel
    point_light: CalibrationModel


def _design(rgb: np.ndarray, fit: str) -> np.ndarray:
    if fit == "linear":
        return rgb
    if fit == "affine":
        return np.column_stack((rgb, np.ones(len(rgb))))
    if fit == "rootpoly2":
        r, g, b = rgb.T

        return np.column_stack(
            (r, g, b, np.sqrt(r * g), np.sqrt(r * b), np.sqrt(g * b), np.ones(len(rgb)))
        )
    raise ValueError(f"unknown fit type: {fit}")


def _decode(values: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "linear":
        return values

    return np.where(
        values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4
    )


def _encode(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.maximum(values, 0) ** (1 / 2.4) - 0.055,
    )


def _lab_to_xyz(lab: np.ndarray, white: np.ndarray) -> np.ndarray:
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    d = 6 / 29

    def f(t):
        return np.where(t > d, t**3, 3 * d * d * (t - 4 / 29))

    return np.column_stack((f(fx), f(fy), f(fz))) * white


def _adapt(xyz: np.ndarray, source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    scale = np.diag((BRADFORD @ destination) / (BRADFORD @ source))
    matrix = np.linalg.solve(BRADFORD, scale @ BRADFORD)

    return (matrix @ xyz.T).T


def fit_model(
    patches_rgb: np.ndarray,
    *,
    exposure_ms: float,
    source_file: str,
    fit_type="affine",
    input_encoding="linear",
    input_scale=255.0,
    reference_white="D65",
) -> CalibrationModel:
    patches = np.asarray(patches_rgb, np.float64)
    if (
        patches.ndim != 2
        or patches.shape[1] != 3
        or len(patches) % 30
        or not np.isfinite(patches).all()
    ):
        raise ValueError(
            "patches must be finite N*30x3 RGB, one 30-patch chart per image"
        )
    normalized = np.clip(patches / input_scale, 0, 1)
    clipped = np.flatnonzero(normalized.max(axis=1) >= 0.9995)
    if clipped.size:
        described = [f"patch {i % 30 + 1} in image {i // 30 + 1}" for i in clipped[:6]]
        more = " and more" if clipped.size > 6 else ""
        warnings.warn(
            f"Clipped highlights (255) at {'; '.join(described)}{more}. "
            "Lower the exposure or move the light, then recapture.",
            stacklevel=2,
        )
    white = D65 if reference_white.upper() == "D65" else D50
    xyz = np.tile(
        _adapt(_lab_to_xyz(REFERENCE_LAB, white), white, D65), (len(patches) // 30, 1)
    )
    coefficients, *_ = np.linalg.lstsq(
        _design(_decode(normalized, input_encoding), fit_type), xyz, rcond=None
    )

    return CalibrationModel(
        1,
        fit_type,
        coefficients.tolist(),
        input_encoding,
        float(input_scale),
        reference_white.upper(),
        "sRGB D65",
        float(exposure_ms),
        "RGB",
        source_file,
        dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def apply_model(
    rgb: np.ndarray, model: CalibrationModel, *, exposure_ms: float
) -> np.ndarray:
    if abs(exposure_ms - model.exposure_ms) > 1e-6:
        raise ValueError(
            f"exposure {exposure_ms} ms does not match calibration exposure {model.exposure_ms} ms"
        )
    values = np.asarray(rgb, np.float64) / model.input_scale
    linear = _decode(np.clip(values, 0, 1), model.input_encoding)
    shape = linear.shape
    xyz = _design(linear.reshape(-1, 3), model.fit_type) @ np.asarray(
        model.coefficients
    )

    return np.clip(_encode(xyz @ XYZ_TO_RGB.T), 0, 1).reshape(shape)


def save_bundle(path: Path, bundle: ColorCalibration) -> None:
    path.write_text(json.dumps(dataclasses.asdict(bundle), indent=2) + "\n")


def load_bundle(path: Path) -> ColorCalibration:
    data = json.loads(path.read_text())

    return ColorCalibration(
        data["version"],
        CalibrationModel(**data["diffuse"]),
        CalibrationModel(**data["point_light"]),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="fabsim colorcal",
        description="Colour calibration from a 30-patch chart: fit and apply per-illumination colour models.",
    )
    p.add_argument(
        "--diffuse-patches",
        type=Path,
        required=True,
        help="(N*30, 3) RGB .npy: the 30 chart patches of each of N images",
    )
    p.add_argument(
        "--point-light-patches",
        type=Path,
        required=True,
        help="(N*30, 3) RGB .npy: the 30 chart patches of each of N images",
    )
    p.add_argument("--diffuse-exposure-ms", type=float, required=True)
    p.add_argument("--point-light-exposure-ms", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    bundle = ColorCalibration(
        1,
        fit_model(
            np.load(args.diffuse_patches),
            exposure_ms=args.diffuse_exposure_ms,
            source_file=args.diffuse_patches.name,
        ),
        fit_model(
            np.load(args.point_light_patches),
            exposure_ms=args.point_light_exposure_ms,
            source_file=args.point_light_patches.name,
        ),
    )
    save_bundle(args.output, bundle)


if __name__ == "__main__":
    main()
