from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image

from ..color_calibration import ColorCalibration, apply_model, save_bundle
from .settings import EXPOSURE_TOLERANCE

ROLE_NOTE = "diffuse -> diffuse model; ps_* and ps_b_* -> point_light model; rgb not corrected"


def basenames(source: str) -> str:
    return "; ".join(PureWindowsPath(part.strip()).name for part in source.split(";") if part.strip())


def corrected_models(stamp: dict | None) -> set[str]:
    both = {"diffuse", "point_light"}
    if not stamp:
        return set()
    if "images_corrected" in stamp:
        return both if stamp["images_corrected"] else set()
    if not stamp.get("applied"):
        return set()
    if "uncorrected" in str(stamp.get("reason", "")):
        return {"diffuse"} if "diffuse corrected post-hoc" in str(stamp.get("roles", "")) else set()

    return both


class ColorCorrector:
    def __init__(
        self,
        bundle: ColorCalibration | None,
        exposure_ms: float,  # the exposure the camera really uses
        keep_raw: bool,
        run_dir: Path,
        log_fn: Callable[[str], None] = lambda _msg: None,
    ):
        self.calibration: ColorCalibration | None = None
        self._bundle_path = run_dir / "color_calibration.json"
        self.exposure_ms = exposure_ms
        self.keep_raw = True
        self.stamp: dict = {
            "images_corrected": False,
            "applied": False,
            "exposure_ms": exposure_ms,
            "roles": ROLE_NOTE,
        }
        if bundle is None:
            self.stamp["reason"] = "no color calibration loaded"

            return
        exposures = {bundle.diffuse.exposure_ms, bundle.point_light.exposure_ms}
        self.stamp.update(
            bundle_version=bundle.version,
            diffuse_model_version=bundle.diffuse.version,
            point_light_model_version=bundle.point_light.version,
            calibration_exposure_ms=sorted(exposures),
            calibration_source=basenames(bundle.diffuse.source_file),
        )
        if any(abs(e - exposure_ms) > EXPOSURE_TOLERANCE * e for e in exposures):
            self.stamp["reason"] = "exposure mismatch, images saved uncorrected"
            log_fn(
                f"WARNING: calibration exposure {sorted(exposures)} ms != run exposure "
                f"{exposure_ms:g} ms; saving raw images without color correction"
            )

            return
        self.calibration = bundle
        self.keep_raw = bool(keep_raw)
        self.stamp.update(
            images_corrected=True,
            applied=True,
            keep_raw=self.keep_raw,
            file="color_calibration.json",
        )
        log_fn(f"Color calibration applied on save (keep raw: {'yes' if self.keep_raw else 'no'})")

    @property
    def applied(self) -> bool:
        return self.calibration is not None

    def save_bundle_copy(self) -> None:
        if self.calibration is None:
            return
        bundle = self.calibration
        clean = dataclasses.replace(
            bundle,
            diffuse=dataclasses.replace(bundle.diffuse, source_file=basenames(bundle.diffuse.source_file)),
            point_light=dataclasses.replace(
                bundle.point_light,
                source_file=basenames(bundle.point_light.source_file),
            ),
        )
        save_bundle(self._bundle_path, clean)

    def model_for_role(self, role: str):
        if self.calibration is None:
            return None
        if role == "diffuse":
            return self.calibration.diffuse
        if role.startswith("ps_"):
            return self.calibration.point_light

        return None

    def correct(self, image: Image.Image, role: str) -> Image.Image | None:
        model = self.model_for_role(role)
        if model is None:
            return None
        # The run exposure was checked against the calibration within the tolerance in __init__.
        corrected = apply_model(np.asarray(image), model, exposure_ms=model.exposure_ms)

        return Image.fromarray((corrected * 255.0 + 0.5).astype(np.uint8))
