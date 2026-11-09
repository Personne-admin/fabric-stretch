import numpy as np
import pytest

from fabsim.color_calibration import (
    CalibrationModel,
    ColorCalibration,
    apply_model,
    fit_model,
    load_bundle,
    main,
    save_bundle,
)


def identity(exposure, source, gain=1.0):
    return CalibrationModel(
        1,
        "linear",
        (np.eye(3) * gain).tolist(),
        "linear",
        1.0,
        "D65",
        "sRGB D65",
        exposure,
        "RGB",
        source,
        "2026-01-01T00:00:00Z",
    )


def test_channels_brightness_and_exposure_are_stable():
    model = identity(15.0, "point.npy")
    rgb = np.array([[[0.2, 0.4, 0.8]]])

    out = apply_model(rgb, model, exposure_ms=15.0)
    assert out[0, 0, 2] > out[0, 0, 1] > out[0, 0, 0]
    assert np.allclose(out, apply_model(rgb, model, exposure_ms=15.0))
    with pytest.raises(ValueError):
        apply_model(
            rgb, model, exposure_ms=25.0
        )  # an exposure mismatch must not be silently scaled


def test_diffuse_and_point_models_round_trip_separately(tmp_path):
    bundle = ColorCalibration(
        1, identity(15.0, "diffuse.npy"), identity(15.0, "point.npy", 0.5)
    )

    path = tmp_path / "calibration.json"
    save_bundle(path, bundle)

    restored = load_bundle(path)
    assert restored == bundle
    assert restored.diffuse.source_file != restored.point_light.source_file

    rgb = np.array([[[0.25, 0.25, 0.25]]])
    assert not np.allclose(
        apply_model(rgb, restored.diffuse, exposure_ms=15.0),
        apply_model(rgb, restored.point_light, exposure_ms=15.0),
    )


def test_clipped_patches_warn_instead_of_printing():
    patches = np.full((30, 3), 120.0)
    patches[4] = 255.0
    with pytest.warns(UserWarning, match="Clipped highlights.*patch 5 in image 1"):
        fit_model(patches, exposure_ms=15.0, source_file="chart.npy")


def test_command_line_stores_file_names_not_paths(tmp_path):
    folder = tmp_path / "private" / "chart"
    folder.mkdir(parents=True)

    rng = np.random.default_rng(0)

    for name in ("diffuse.npy", "point.npy"):
        np.save(folder / name, 20 + 200 * rng.random((60, 3)))  # two charts per array

    out = tmp_path / "calibration.json"
    main(
        ["--diffuse-patches", str(folder / "diffuse.npy"), "--point-light-patches", str(folder / "point.npy"),
         "--diffuse-exposure-ms", "15", "--point-light-exposure-ms", "15", "--output", str(out)]
    )  # fmt: skip

    bundle = load_bundle(out)
    assert (bundle.diffuse.source_file, bundle.point_light.source_file) == (
        "diffuse.npy",
        "point.npy",
    )
    assert "private" not in out.read_text()
