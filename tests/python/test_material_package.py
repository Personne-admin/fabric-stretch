from __future__ import annotations

import json
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

import synthetic_project as syn
from fabsim import dataset, tiling
from fabsim import photometric as ph
from fabsim.material_package import (
    ForceTare,
    Tile,
    build,
    clip_threshold,
    fit_tare_offset,
    force_quality,
    grid_coordinates,
    grid_frames,
    register_references,
    remap_grid,
    remap_shared,
    spearman,
    stretch_ratio,
    tared_force,
)
from fabsim.material_package import main as package_main


def _srgb_to_linear(rgb):
    x = np.asarray(rgb, dtype=np.float32) / 255

    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb):
    x = np.clip(rgb, 0, 1)

    return np.clip(
        np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055) * 255 + 0.5,
        0,
        255,
    ).astype(np.uint8)


def interpolate(package_dir, branch, stretch):
    meta = json.loads((package_dir / "package.json").read_text())

    states = sorted(meta["branches"][branch], key=lambda s: s["stretch_ratio"])
    if (
        stretch < states[0]["stretch_ratio"] - 1e-6
        or stretch > states[-1]["stretch_ratio"] + 1e-6
    ):
        raise ValueError(f"stretch {stretch} is outside measured {branch} support")

    lo = max(
        (s for s in states if s["stretch_ratio"] <= stretch),
        key=lambda s: s["stretch_ratio"],
        default=states[0],
    )
    hi = min(
        (s for s in states if s["stretch_ratio"] >= stretch),
        key=lambda s: s["stretch_ratio"],
        default=states[-1],
    )

    a = np.asarray(Image.open(package_dir / lo["texture"]).convert("RGB"))
    if lo is hi:
        return a

    b = np.asarray(Image.open(package_dir / hi["texture"]).convert("RGB"))
    t = (stretch - lo["stretch_ratio"]) / (hi["stretch_ratio"] - lo["stretch_ratio"])

    return _linear_to_srgb((1 - t) * _srgb_to_linear(a) + t * _srgb_to_linear(b))


def press_path(span_mm, width_mm, depth_mm, center_mm=None):
    if center_mm is None:
        center_mm = span_mm / 2
    if (
        not all(np.isfinite(v) for v in (span_mm, width_mm, depth_mm, center_mm))
        or span_mm <= 0
        or width_mm < 0
        or width_mm >= span_mm
        or depth_mm < 0
        or center_mm - width_mm / 2 < 0
        or center_mm + width_mm / 2 > span_mm
    ):
        raise ValueError("invalid press geometry")

    left = center_mm - width_mm / 2
    right = span_mm - center_mm - width_mm / 2
    length = np.hypot(left, depth_mm) + width_mm + np.hypot(right, depth_mm)

    return {
        "stretch_ratio": float(length / span_mm),
        "left_length_mm": float(np.hypot(left, depth_mm)),
        "right_length_mm": float(np.hypot(right, depth_mm)),
    }


def frame(dx=0, scale=1, bad=None):
    return {
        r * 4
        + c: {
            "x": 10 + c * 10 * scale + dx,
            "y": 10 + r * 10,
            "confidence": 0 if (r, c) == bad else 1,
        }
        for r in range(3)
        for c in range(4)
    }


def test_inverse_registration_removes_translation_and_stretch():
    t = Tile(0, 0, 3, 2, (10, 10, 40, 30), (31, 21))
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    source[:, :, :] = np.arange(100, dtype=np.uint8)[None, :, None]
    a = remap_grid(source, frame(), 4, t)

    assert a.shape == (21, 31, 3)
    assert int(a[10, 10, 0]) == 20
    assert int(remap_grid(source, frame(dx=5), 4, t)[10, 10, 0]) == 25
    assert int(remap_grid(source, frame(scale=2), 4, t)[10, 10, 0]) == 30


def test_mapping_outside_the_image_is_refused():
    t = Tile(0, 0, 3, 2, (10, 10, 40, 30), (31, 21))
    with pytest.raises(ValueError, match="leaves image support"):
        remap_grid(np.zeros((30, 30, 3), np.uint8), frame(), 4, t)


def test_shared_atlas_tracks_same_physical_pixels_across_branches():
    image = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, 3]]), (256, 256))

    H = register_references(image, shifted)
    tile = Tile(0, 0, 3, 2, (10, 10, 40, 30), (31, 21))

    yy, xx = np.meshgrid(np.linspace(0, 2, 21), np.linspace(0, 3, 31), indexing="ij")
    canonical = np.stack((10 + 10 * xx, 10 + 10 * yy), axis=-1).astype(np.float32)
    release_xy = cv2.perspectiveTransform(canonical.reshape(1, -1, 2), H).reshape(
        canonical.shape
    )

    release_grid = grid_coordinates(frame(dx=5), 4, 3, release_xy)
    a = remap_grid(image, frame(), 4, tile)
    b = remap_shared(shifted, frame(dx=5), 4, 3, release_grid)
    assert np.abs(a.astype(float) - b.astype(float)).mean() < 12


def test_stretch_reference_and_known_extension():
    t = Tile(0, 0, 3, 2, (10, 10, 40, 30), (31, 21))
    assert stretch_ratio(frame(), frame(), 4, t, "horizontal") == pytest.approx(1)
    assert stretch_ratio(
        frame(scale=1.1), frame(), 4, t, "horizontal"
    ) == pytest.approx(1.1)
    assert stretch_ratio(frame(), frame(), 4, t, "vertical") == pytest.approx(1)


def test_linear_interpolation_and_branch_separation(tmp_path):
    for branch, values in [("loading", [0, 255]), ("release", [80, 160])]:
        (tmp_path / branch).mkdir()
        for i, value in enumerate(values):
            Image.new("RGB", (2, 2), (value,) * 3).save(tmp_path / branch / f"{i}.png")

    meta = {
        "branches": {
            b: [
                {"stretch_ratio": 1 + i * 0.1, "texture": f"{b}/{i}.png"}
                for i in range(2)
            ]
            for b in ("loading", "release")
        }
    }

    (tmp_path / "package.json").write_text(json.dumps(meta))
    assert int(interpolate(tmp_path, "loading", 1)[0, 0, 0]) == 0
    assert int(interpolate(tmp_path, "loading", 1.05)[0, 0, 0]) == 188
    assert int(interpolate(tmp_path, "release", 1)[0, 0, 0]) == 80
    with pytest.raises(ValueError):
        interpolate(tmp_path, "loading", 2)


def test_press_geometry():
    assert press_path(200, 40, 0)["stretch_ratio"] == 1
    assert press_path(200, 40, 20)["stretch_ratio"] > 1
    assert (
        press_path(200, 40, 20)["left_length_mm"]
        == press_path(200, 40, 20)["right_length_mm"]
    )
    assert (
        press_path(200, 80, 20)["stretch_ratio"]
        > press_path(200, 40, 20)["stretch_ratio"]
    )
    for args in [(0, 20, 2), (200, 201, 2), (200, 20, -1)]:
        with pytest.raises(ValueError):
            press_path(*args)


@pytest.mark.realdata
def test_real_tracker_topology(data_dir):
    tracks = data_dir / "fabric_002/warp_loading/tracks.json"
    if not tracks.is_file():
        pytest.skip("fabric_002 tracks not available")
    frames, ref, cols, rows = grid_frames(tracks)
    assert ref == 0
    assert cols * rows == len(frames[0])


TILING = tiling.TilingConfig(window_size=(8, 8), shift=2, interp_dist=1)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    syn.make_project(root)
    tiled, untiled = root / "material_package_v3", root / "material_package_v3_untiled"
    report = build(root, tiled, untiled, TILING)

    return {"root": root, "tiled": tiled, "untiled": untiled, "report": report}


def _package(directory, orientation="warp"):
    return json.loads((directory / orientation / "package.json").read_text())


def _read(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, path

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def test_orientation_discovery_and_no_empty_directories(built):
    entry = built["report"]["orientations"]
    assert entry["warp"]["status"] == "built"
    assert entry["weft"]["status"] == "skipped"
    assert "release" in entry["weft"]["reason"]
    for root in (built["tiled"], built["untiled"]):
        assert not (root / "weft").exists()
        assert (
            json.loads((root / "build_report.json").read_text())["orientations"][
                "weft"
            ]["status"]
            == "skipped"
        )
        for directory in [root, *(d for d in root.rglob("*") if d.is_dir())]:
            assert any(directory.iterdir()), f"empty directory {directory}"


def test_v3_metadata(built):
    for directory in (built["tiled"], built["untiled"]):
        pkg = _package(directory)
        assert (pkg["format"], pkg["version"], pkg["orientation"]) == (
            "fabsim-material-rgb",
            3,
            "warp",
        )
        assert pkg["atlas"]["active_grid_axis"] == "horizontal"
        assert pkg["force"]["unit"] == "N"
        assert pkg["force"]["clamp_width_m"] == syn.CLAMP_WIDTH_M
        assert pkg["force"]["zero_counts"] == {
            "loading": syn.LOADING_ZERO,
            "release": syn.RELEASE_ZERO,
        }
        assert pkg["ps"]["albedo_scale"] > 0
        assert len(pkg["ps"]["gains"]) == 3
        assert "release_residual_stretch" in pkg
        for branch in ("loading", "release"):
            states = pkg["branches"][branch]
            assert [s["sequence_index"] for s in states] == list(range(len(states)))
            assert set(states[0]) >= {
                "force_n",
                "force_raw_n",
                "stretch_ratio",
                "texture",
                "albedo",
                "normal",
            }
    assert "tiling" in _package(built["tiled"])
    assert "tiling" not in _package(built["untiled"])


def test_force_is_tared_at_the_loading_reference_and_looked_up_by_index(built):
    pkg = _package(built["untiled"])
    loading, release = pkg["branches"]["loading"], pkg["branches"]["release"]

    assert pkg["force"]["offset_n"] == syn.LOADING_FORCE[0]
    assert [s["force_raw_n"] for s in loading] == syn.LOADING_FORCE
    assert [s["force_raw_n"] for s in release] == pytest.approx(
        [syn.stored_force("release", i) for i in range(4)]
    )

    assert loading[0]["force_n"] == 0
    for state, raw in zip(loading, syn.LOADING_FORCE, strict=True):
        assert state["force_n"] == pytest.approx(raw - syn.LOADING_FORCE[0])

    for state, force in zip(release, syn.RELEASE_FORCE, strict=True):
        assert state["force_n"] == pytest.approx(force - syn.LOADING_FORCE[0])

    # the frame without a valid force reading is left out and reported
    assert len(loading) == len(syn.LOADING_FORCE)
    assert any(
        "no valid force" in note
        for note in built["report"]["orientations"]["warp"]["skipped_states"]
    )


def test_release_stretch_is_measured_against_the_loading_reference(built):
    pkg = _package(built["untiled"])
    loading = [s["stretch_ratio"] for s in pkg["branches"]["loading"]]
    release = [s["stretch_ratio"] for s in pkg["branches"]["release"]]

    assert loading == pytest.approx(
        [1 + syn.STEP * i for i in range(len(loading))], abs=0.004
    )

    assert pkg["release_residual_stretch"] == pytest.approx(
        syn.RELEASE_RESIDUAL, abs=0.005
    )

    assert release[-1] == pkg["release_residual_stretch"] > 1.02
    assert release[0] == pytest.approx(
        loading[-1], abs=0.005
    )  # release starts where loading ended

    assert release == sorted(release, reverse=True)


def test_photometric_layers(built):
    pkg = _package(built["untiled"])
    width, height = pkg["atlas"]["dimensions_px"]
    scale = pkg["ps"]["albedo_scale"]
    assert pkg["ps"]["gains"] == pytest.approx(
        list(syn.GAINS / syn.GAINS.mean()), abs=0.03
    )

    for branch in ("loading", "release"):
        for state in pkg["branches"][branch]:
            rgb_path = built["untiled"] / "warp" / state["texture"]
            raw = cv2.imread(
                str(built["untiled"] / "warp" / state["albedo"]), cv2.IMREAD_UNCHANGED
            )

            assert raw.dtype == np.uint16 and raw.shape == (height, width, 3)
            normal = cv2.imread(
                str(built["untiled"] / "warp" / state["normal"]), cv2.IMREAD_UNCHANGED
            )

            assert normal.dtype == np.uint16 and normal.shape == (height, width, 3)

            rgb = _read(rgb_path)
            assert rgb.dtype == np.uint8 and rgb.shape == (height, width, 3)

            # the scene is flat and gray: albedo equals the diffuse image, normals face the camera
            albedo = ph.decode_albedo_u16(
                _read(built["untiled"] / "warp" / state["albedo"]), scale
            )
            assert np.abs(albedo - rgb / 255).mean() < 3 / 255

            n = ph.decode_normals_u16(
                _read(built["untiled"] / "warp" / state["normal"])
            )
            assert np.abs(np.linalg.norm(n, axis=-1) - 1).max() < 1e-3
            assert np.median(n, axis=(0, 1)) == pytest.approx([0, 0, 1], abs=0.02)


def test_tiled_layers_use_the_same_area_and_seams_as_rgb(built):
    pkg = _package(built["tiled"])
    meta = pkg["tiling"]
    tile_w, tile_h = meta["tile_size_px"]
    seams = tiling.Seams.load(built["tiled"] / "warp" / "tiling_seams.npz")

    assert seams.atlas_size == tuple(pkg["atlas"]["dimensions_px"])
    assert (seams.area[2] - 1, seams.area[0] - 1) == tuple(meta["origin_px"])

    scale = pkg["ps"]["albedo_scale"]
    for branch in ("loading", "release"):
        for state in pkg["branches"][branch]:
            untiled = {
                k: _read(built["untiled"] / "warp" / state[k])
                for k in ("texture", "albedo", "normal")
            }

            tiled = {k: _read(built["tiled"] / "warp" / state[k]) for k in untiled}
            for key, image in tiled.items():
                assert image.shape[:2] == (tile_h, tile_w), key
                assert image.dtype == untiled[key].dtype
            # all layers were cut with the stored seams: RGB exactly, the others identically to apply_tiling

            assert np.array_equal(
                tiled["texture"], tiling.apply_tiling(untiled["texture"], seams)
            )
            assert np.array_equal(
                tiled["albedo"], tiling.apply_tiling(untiled["albedo"], seams)
            )
            assert np.array_equal(
                tiled["normal"],
                tiling.apply_tiling(untiled["normal"], seams, normals=True),
            )

            # and they still describe the same pixels: albedo equals RGB (flat gray cloth), normals stay unit
            albedo = ph.decode_albedo_u16(tiled["albedo"], scale)
            assert np.abs(albedo - tiled["texture"] / 255).mean() < 3 / 255
            n = ph.decode_normals_u16(tiled["normal"])
            assert np.abs(np.linalg.norm(n, axis=-1) - 1).max() < 1e-3


def test_untiled_build_is_independent_of_tiling(built, tmp_path):
    only_untiled = build(built["root"], None, tmp_path / "u", None)
    assert only_untiled["orientations"]["warp"]["status"] == "built"
    for branch in ("loading", "release"):
        a = _package(tmp_path / "u")["branches"][branch]
        b = _package(built["untiled"])["branches"][branch]
        assert a == b


def test_existing_output_needs_force(built, tmp_path):
    out = tmp_path / "u"
    build(built["root"], None, out, None, ps=False)
    with pytest.raises(FileExistsError):
        build(built["root"], None, out, None, ps=False)

    with pytest.raises(SystemExit):
        package_main(
            [str(built["root"]), "--no-tiling", "--no-ps", "--untiled-out", str(out)]
        )

    stale = out / "warp" / "stale.txt"
    stale.write_text("old")
    assert (
        package_main(
            [
                str(built["root"]),
                "--no-tiling",
                "--no-ps",
                "--untiled-out",
                str(out),
                "--force",
            ]
        )
        == 0
    )

    assert not stale.exists()
    pkg = _package(out)
    assert "ps" not in pkg and "albedo" not in pkg["branches"]["loading"][0]
    assert not list((out / "warp").rglob("albedo_*"))


def test_cli_defaults_and_orientation_filter(tmp_path):
    root = syn.make_project(tmp_path / "p", orientations=("warp", "bias"), ps=False)
    code = package_main(
        [str(root), "--orientations", "bias", "--tiling-window-size", "8", "8", "--tiling-shift", "2",
         "--tiling-interp-dist", "1"]
    )  # fmt: skip

    assert code == 0
    for name in ("material_package_v3", "material_package_v3_untiled"):
        assert (root / name / "bias" / "package.json").is_file()
        assert not (root / name / "warp").exists()

    assert (root / "material_package_v3" / "bias" / "tiling_seams.npz").is_file()


def test_tiling_that_does_not_fit_leaves_the_orientation_untiled(tmp_path):
    root = syn.make_project(tmp_path / "p", ps=False)
    report = build(
        root,
        tmp_path / "t",
        tmp_path / "u",
        tiling.TilingConfig(window_size=(1000, 1000)),
    )

    entry = report["orientations"]["warp"]
    assert "tiling_skipped" in entry
    assert not (tmp_path / "t" / "warp").exists()
    assert (tmp_path / "u" / "warp" / "package.json").is_file()


def _rewrite_force(run, zero_counts):
    path = run / "force.json"
    doc = json.loads(path.read_text())
    for entry in doc["forces"]:
        if entry["status"] == "ok":
            entry["zero_counts"] = zero_counts
            entry["force_n"] = (zero_counts - entry["raw_counts"]) * entry[
                "calibration_n_per_count"
            ]

    path.write_text(json.dumps(doc))


def test_tare_depends_on_the_counts_not_on_the_stored_zero(tmp_path):
    reference = syn.make_project(
        tmp_path / "a", ps=False
    )  # Release stored with zero -279

    other = syn.make_project(tmp_path / "b", ps=False)
    _rewrite_force(
        other / "warp_release", syn.LOADING_ZERO
    )  # the very same counts, Loading zero

    states = {}
    for name, root in (("a", reference), ("b", other)):
        build(root, None, tmp_path / name / "out", None, ps=False)
        states[name] = _package(tmp_path / name / "out")["branches"]["release"]

    assert [s["force_n"] for s in states["a"]] == pytest.approx(
        [s["force_n"] for s in states["b"]]
    )

    assert states["a"][0]["force_raw_n"] != pytest.approx(states["b"][0]["force_raw_n"])


def test_failed_orientation_leaves_no_directory(tmp_path):
    root = syn.make_project(tmp_path / "p")
    ps_image = root / "warp_release" / "images" / "img_0002_ps_2.png"
    data = ps_image.read_bytes()
    ps_image.write_bytes(
        data[:48] + data[-12:]
    )  # valid header and end chunk, broken data in between

    tiled, untiled = tmp_path / "t", tmp_path / "u"
    report = build(root, tiled, untiled, TILING, diagnostics_out=tmp_path / "d")

    assert report["orientations"]["warp"]["status"] == "skipped"
    assert "cannot read image" in report["orientations"]["warp"]["reason"]
    for directory in (tiled, untiled, tmp_path / "d"):
        assert not (directory / "warp").exists()


def test_force_replaces_the_previous_package_of_a_failing_orientation(tmp_path):
    root = syn.make_project(tmp_path / "p", ps=False)
    out = tmp_path / "u"
    assert (
        build(root, None, out, None, ps=False)["orientations"]["warp"]["status"]
        == "built"
    )

    (root / "warp_release" / "tracks.json").write_text(
        "{}"
    )  # now the orientation fails

    report = build(root, None, out, None, ps=False, force=True)

    assert report["orientations"]["warp"]["status"] == "skipped"
    assert not (
        out / "warp"
    ).exists()  # the old package must not survive next to a "skipped" report


def test_ps_images_of_the_wrong_size_leave_an_rgb_only_package(tmp_path):
    root = syn.make_project(tmp_path / "p")

    ps_image = root / "warp_loading" / "images" / "img_0001_ps_1.png"
    cv2.imwrite(str(ps_image), np.zeros((100, 100), np.uint8))
    report = build(root, None, tmp_path / "u", None)

    entry = report["orientations"]["warp"]
    assert entry["status"] == "built" and "differ in size" in entry["ps_skipped"]

    pkg = _package(tmp_path / "u")
    assert "ps" not in pkg and "albedo" not in pkg["branches"]["loading"][0]
    assert not list((tmp_path / "u" / "warp").rglob("albedo_*"))


def test_diagnostics_are_opt_in(tmp_path):
    root = syn.make_project(tmp_path / "p", ps=False)

    build(root, tmp_path / "t", tmp_path / "u", TILING, ps=False)
    assert not list(tmp_path.rglob("*preview_2x2.png"))

    build(
        root,
        tmp_path / "t2",
        tmp_path / "u2",
        TILING,
        ps=False,
        diagnostics_out=tmp_path / "d",
    )
    assert (tmp_path / "d" / "warp" / "warp_preview_2x2.png").is_file()
    assert not (tmp_path / "t2" / "diagnostics").exists()


def test_tiling_guidance_out_of_range_keeps_the_untiled_package(tmp_path):
    root = syn.make_project(tmp_path / "p", ps=False)
    config = tiling.TilingConfig(
        window_size=(8, 8), shift=2, interp_dist=1, guidance=[0, 99]
    )

    report = build(root, tmp_path / "t", tmp_path / "u", config, ps=False)
    entry = report["orientations"]["warp"]

    assert entry["status"] == "built" and "out of range" in entry["tiling_skipped"]
    assert (tmp_path / "u" / "warp" / "package.json").is_file()
    assert not (tmp_path / "t" / "warp").exists()


def test_negative_tiling_guidance_is_a_usage_error(tmp_path):
    root = syn.make_project(tmp_path / "p", ps=False)
    with pytest.raises(SystemExit):
        package_main([str(root), "--tiling-guidance", "-1", "--no-ps"])

    assert not (root / "material_package_v3").exists()


def test_session_led_snapshot_wins_over_led_config_file(tmp_path):
    root = syn.make_project(tmp_path / "p", orientations=("warp", "bias"))
    snapshot = ph.load_light_config()
    session = root / "warp_loading" / "session.json"
    doc = json.loads(session.read_text())
    doc["led_config"] = {
        "light_azimuths_deg": list(snapshot.azimuths_deg),
        "light_elevation_deg": snapshot.elevation_deg,
    }

    session.write_text(json.dumps(doc))
    led = tmp_path / "other_led.json"
    led.write_text(
        json.dumps({"light_azimuths_deg": [0, 120, 240], "light_elevation_deg": 30})
    )

    build(
        root, None, tmp_path / "u", None, led_config=led, orientations=["warp", "bias"]
    )

    assert (
        _package(tmp_path / "u", "warp")["ps"]["light"]["source"]
        == "session.json:led_config"
    )
    assert (
        _package(tmp_path / "u", "bias")["ps"]["light"]["source"]
        == "file:other_led.json"
    )


def test_albedo_scale_covers_every_state(tmp_path):
    root = syn.make_project(tmp_path / "p", brighten=0.3)
    report = build(root, tmp_path / "t", tmp_path / "u", TILING)
    entry = report["orientations"]["warp"]

    pkg = _package(tmp_path / "u")
    scale = pkg["ps"]["albedo_scale"]
    assert "all 8 states" in pkg["ps"]["albedo_scale_basis"]

    reference = _read(tmp_path / "u" / "warp" / pkg["branches"]["loading"][0]["albedo"])
    assert clip_threshold(ph.decode_albedo_u16(reference, scale)) < 0.85 * scale

    for root_dir in (tmp_path / "u", tmp_path / "t"):
        for branch in ("loading", "release"):
            for state in _package(root_dir)["branches"][branch]:
                raw = cv2.imread(
                    str(root_dir / "warp" / state["albedo"]), cv2.IMREAD_UNCHANGED
                )
                assert (raw == 65535).any(axis=-1).mean() < 0.001, (
                    root_dir.name,
                    branch,
                    state["frame_id"],
                )

    assert max(entry["ps"]["clipped_fraction_max"].values()) < 0.001
    assert set(entry["ps"]["clipped_fraction_max"]) == {"untiled", "tiled"}


def test_run_folders_with_other_names_are_reported(tmp_path):
    root = syn.make_project(tmp_path / "p", ps=False)
    shutil.copytree(
        root / "warp_loading", root / "warp_loading_2"
    )  # the rig's collision suffix

    shutil.copytree(root / "warp_release", root / "bias_run")
    (root / "weft_release").mkdir()
    (root / "weft_release" / "session.json").write_text(
        "{}"
    )  # right name, but no frames.json

    (root / "notes").mkdir()

    ignored = dict(dataset.ignored_run_folders(root))
    assert set(ignored) == {"warp_loading_2", "bias_run", "weft_release"}
    assert "rename it to replace warp_loading" in ignored["warp_loading_2"]
    assert ignored["weft_release"] == "no frames.json"

    report = build(root, None, tmp_path / "u", None, ps=False)
    assert {i["name"] for i in report["ignored_run_folders"]} == set(ignored)
    assert (
        tmp_path / "u" / "warp" / "package.json"
    ).is_file()  # warp_loading_2 did not replace anything

    assert (
        package_main(
            [str(root), "--no-tiling", "--no-ps", "--untiled-out", str(tmp_path / "v")]
        )
        == 0
    )


def test_a_project_without_odd_folders_reports_none(tmp_path):
    root = syn.make_project(tmp_path / "p", ps=False)
    assert (
        build(root, None, tmp_path / "u", None, ps=False).get("ignored_run_folders")
        is None
    )


def _layers(package_dir, state, orientation="warp"):
    pkg = _package(package_dir, orientation)
    root = package_dir / orientation
    albedo = ph.decode_albedo_u16(
        _read(root / state["albedo"]), pkg["ps"]["albedo_scale"]
    )

    normal = ph.decode_normals_u16(_read(root / state["normal"]))

    return albedo, normal, _read(root / state["texture"]).astype(np.float32) / 255


def test_head_on_relight_is_as_bright_as_the_diffuse_photo(tmp_path):
    root = syn.make_project(tmp_path / "p", diffuse_gain=1.2)
    build(root, None, tmp_path / "u", None)

    pkg = _package(tmp_path / "u")
    assert pkg["ps"]["exposure_gain"] == pytest.approx(1.2, rel=0.02)
    assert "exposure_gain" in pkg["ps"]["exposure_basis"]

    for branch in ("loading", "release"):
        for state in pkg["branches"][branch]:
            albedo, normal, rgb = _layers(tmp_path / "u", state)

            head_on = ph.shade(
                albedo, normal, ph.light_direction(0, 90), intensity=1.0, ambient=0.0
            )
            assert head_on.mean() == pytest.approx(rgb.mean(), rel=0.03)

            default = ph.shade(
                albedo, normal, ph.light_direction(135, 45), intensity=1.0, ambient=0.1
            )
            assert 0.6 * rgb.mean() < default.mean() < rgb.mean()

    # LED i is reproduced with intensity gains[i] / exposure_gain: compare with the captured photograph
    state = pkg["branches"]["loading"][0]
    albedo, normal, _ = _layers(tmp_path / "u", state)
    x0, y0, x1, y1 = (int(round(v)) for v in pkg["atlas"]["reference_bounds_px"])
    directions = ph.light_matrix(ph.load_light_config())
    for i, key in enumerate(dataset.PS_KEYS):
        shot = cv2.imread(
            str(root / "warp_loading" / "images" / f"img_0000_ps_{i + 1}.png"),
            cv2.IMREAD_GRAYSCALE,
        )

        captured = shot[y0:y1, x0:x1].astype(np.float32).mean() / 255
        intensity = pkg["ps"]["gains"][i] / pkg["ps"]["exposure_gain"]
        shown = (
            ph.shade(albedo, normal, directions[i], intensity=intensity)
            @ ph.LUMA
            / ph.LUMA.sum()
        )
        assert shown.mean() == pytest.approx(captured, rel=0.03), key


def test_linear_tilt_of_the_normals_is_removed_in_both_packages(tmp_path):
    root = syn.make_project(tmp_path / "p", ramp=0.3)
    build(root, tmp_path / "t", tmp_path / "u", TILING)
    for name in ("t", "u"):
        pkg = _package(tmp_path / name)
        for branch in ("loading", "release"):
            for state in pkg["branches"][branch]:
                _, normal, _ = _layers(tmp_path / name, state)

                columns = normal[..., 0].mean(axis=0)
                assert abs(columns[:8].mean() - columns[-8:].mean()) < 0.03, (
                    name,
                    branch,
                    state["frame_id"],
                )
                assert ph.mean_normal_tilt_deg(normal) < 0.5

    assert (
        max(
            s["normal_tilt_deg"]
            for s in _package(tmp_path / "u")["branches"]["loading"]
        )
        > 1.0
    )
    assert _package(tmp_path / "u")["ps"]["quality"]["normal_tilt_deg"]["max"] > 1.0
    assert "plane" in _package(tmp_path / "u")["ps"]["normal_correction"]


def test_gains_are_estimated_for_every_state(tmp_path):
    root = syn.make_project(tmp_path / "p", ps=True, gain_drift=0.3)
    build(root, None, tmp_path / "per", None)
    build(root, None, tmp_path / "fixed", None, ps_gains="reference")

    per, fixed = _package(tmp_path / "per"), _package(tmp_path / "fixed")
    assert (
        per["ps"]["gains_mode"] == "per-state" and fixed["ps"]["gains_mode"] == "fixed"
    )

    loading = per["branches"]["loading"]
    assert loading[-1]["ps_gains"][1] < 0.85 * loading[0]["ps_gains"][1]
    assert max(s["normal_tilt_deg"] for b in per["branches"].values() for s in b) < 1.0
    assert (
        max(s["normal_tilt_deg"] for b in fixed["branches"].values() for s in b) > 3.0
    )

    with pytest.raises(ValueError, match="ps_gains"):
        build(root, None, tmp_path / "x", None, ps_gains="sometimes")


def test_gains_of_the_led_configuration_are_never_re_estimated(tmp_path):
    root = syn.make_project(tmp_path / "p", gain_drift=0.3)

    session = root / "warp_loading" / "session.json"
    doc = json.loads(session.read_text())
    cfg = ph.load_light_config()
    doc["led_config"] = {
        "light_azimuths_deg": list(cfg.azimuths_deg),
        "light_elevation_deg": cfg.elevation_deg,
        "gains": [1.0, 1.0, 1.0],
    }

    session.write_text(json.dumps(doc))
    build(root, None, tmp_path / "u", None)
    pkg = _package(tmp_path / "u")
    assert pkg["ps"]["gains_mode"] == "fixed" and pkg["ps"]["gains"] == [1.0, 1.0, 1.0]
    assert all(
        s["ps_gains"] == [1.0, 1.0, 1.0] for b in pkg["branches"].values() for s in b
    )


def _cut_short(path):
    path.write_bytes(path.read_bytes()[:-30])  # a copy that stopped early: no end chunk


def test_a_state_with_an_unreadable_ps_image_is_left_out_of_the_package(tmp_path):
    root = syn.make_project(tmp_path / "p")
    _cut_short(root / "warp_release" / "images" / "img_0002_ps_2.png")

    report = build(root, tmp_path / "t", tmp_path / "u", TILING)
    entry = report["orientations"]["warp"]
    assert entry["status"] == "built" and "ps_skipped" not in entry
    assert entry["dropped_states"] == [
        {
            "branch": "release",
            "frame_id": 2,
            "reason": entry["dropped_states"][0]["reason"],
        }
    ]
    assert "img_0002_ps_2.png" in entry["dropped_states"][0]["reason"]
    assert any("frame 2 left out" in w for w in entry["warnings"])
    for directory in (tmp_path / "t", tmp_path / "u"):
        pkg = _package(directory)
        assert [s["frame_id"] for s in pkg["branches"]["release"]] == [0, 1, 3]
        assert len(pkg["branches"]["loading"]) == 4
        assert all(
            "albedo" in s and "normal" in s for b in pkg["branches"].values() for s in b
        )
        assert not list((directory / "warp" / "release").glob("*0002*"))


def test_unreadable_ps_image_at_a_reference_frame_gives_an_rgb_only_package(tmp_path):
    root = syn.make_project(tmp_path / "p")
    _cut_short(
        root / "warp_loading" / "images" / "img_0000_ps_1.png"
    )  # the tracker reference frame

    report = build(root, None, tmp_path / "u", None)
    entry = report["orientations"]["warp"]
    assert entry["status"] == "built" and "reference frame 0" in entry["ps_skipped"]
    assert "ps" not in _package(tmp_path / "u")
    assert not list((tmp_path / "u" / "warp").rglob("albedo_*"))


def test_a_poor_albedo_correlation_is_reported(tmp_path):
    root = syn.make_project(tmp_path / "p", invert_diffuse=True)

    report = build(root, None, tmp_path / "u", None)
    entry = report["orientations"]["warp"]
    quality = _package(tmp_path / "u")["ps"]["quality"]

    assert quality["albedo_diffuse_correlation"] < 0
    assert quality["warnings"] and "correlate poorly" in quality["warnings"][0]
    assert any("correlate poorly" in w for w in entry["warnings"])

    good = build(syn.make_project(tmp_path / "q"), None, tmp_path / "v", None)
    assert good["orientations"]["warp"]["warnings"] == []
    assert _package(tmp_path / "v")["ps"]["quality"]["albedo_diffuse_correlation"] > 0.9


def test_a_release_that_ends_below_the_loading_reference_is_reported(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(syn, "RELEASE_RESIDUAL", 0.99)
    root = syn.make_project(tmp_path / "p", ps=False)
    entry = build(root, None, tmp_path / "u", None, ps=False)["orientations"]["warp"]
    assert entry["release_residual_stretch"] < 1
    assert any("below 1" in w for w in entry["warnings"])


def test_tare_offset_is_the_intercept_of_a_line_through_the_relaxed_states():
    x = 1 + 0.003 * np.arange(10)
    noise = np.array([1.2, -0.9, 1.5, -1.1, 0.8, -1.4, 1.0, -0.7, 1.3, -1.0])

    force = 2.0 + 60.0 * (x - 1) + noise
    assert fit_tare_offset(x, force) == pytest.approx(2.0, abs=0.5)
    assert abs(fit_tare_offset(x, force) - 2.0) < abs(
        force[0] - 2.0
    )  # better than one sample

    assert (
        fit_tare_offset([1.0, 1.0, 1.0], [1.0, 2.0, 9.0]) == 2.0
    )  # no stretch range: median

    assert fit_tare_offset([1.0], [3.5]) == 3.5


def test_tare_fit_covers_the_first_loading_states(tmp_path, monkeypatch):
    count = 12
    monkeypatch.setattr(syn, "STEP", 0.004)
    rng = np.random.default_rng(4)
    stretch = 1 + 0.004 * np.arange(count)
    monkeypatch.setattr(
        syn,
        "LOADING_FORCE",
        list(-4.0 + 50.0 * (stretch - 1) + rng.normal(0, 1.0, count)),
    )

    monkeypatch.setattr(syn, "RELEASE_FORCE", list(np.linspace(1.5, 0.1, count)))
    root = tmp_path / "p"
    for branch in ("loading", "release"):
        syn.write_run(root / f"warp_{branch}", branch, count=count, ps=False)

    build(root, None, tmp_path / "fit", None, ps=False)
    build(root, None, tmp_path / "one", None, ps=False, tare_states=1)
    fitted, single = _package(tmp_path / "fit"), _package(tmp_path / "one")
    loading = fitted["branches"]["loading"]
    window = [s for s in loading if s["stretch_ratio"] <= 1.03][:10]
    assert fitted["force"]["tare_states"] == len(window) >= 3
    expected = fit_tare_offset(
        [s["stretch_ratio"] for s in window], [s["force_raw_n"] for s in window]
    )

    assert fitted["force"]["offset_n"] == pytest.approx(expected)
    assert fitted["force"]["offset_n"] != pytest.approx(loading[0]["force_raw_n"])
    assert loading[0]["force_n"] == pytest.approx(loading[0]["force_raw_n"] - expected)

    # one state: the reference reading itself, F(reference) = 0 exactly
    assert single["force"]["tare_states"] == 1
    assert single["branches"]["loading"][0]["force_n"] == 0


def test_zero_counts_of_another_run_are_re_referenced_to_the_loading_zero():
    per_count = 0.0125
    # the same raw counts read 349 counts * 0.0125 N = 4.3625 N lower with the stale zero -628
    run = type(
        "Run",
        (),
        {
            "forces": {
                7: {
                    "force_n": 1.0,
                    "zero_counts": -279,
                    "calibration_n_per_count": per_count,
                }
            }
        },
    )

    tare = ForceTare(0.5, {"loading": -628, "release": -279})
    assert tare.zero_counts_differ

    force_n, raw = tared_force(run, {"force_json_index": 7}, tare)
    assert raw == 1.0
    assert force_n == pytest.approx(1.0 + (-628 + 279) * per_count - 0.5)

    same = ForceTare(0.5, {"loading": -279, "release": -279})
    assert not same.zero_counts_differ
    assert tared_force(run, {"force_json_index": 7}, same)[0] == pytest.approx(0.5)

    unknown = type(
        "Run", (), {"forces": {7: {"force_n": 1.0}}}
    )  # no counts recorded: nothing to shift
    assert tared_force(unknown, {"force_json_index": 7}, tare)[0] == pytest.approx(0.5)


def _branch(force, stretch=None):
    stretch = np.linspace(1, 1.17, len(force)) if stretch is None else stretch

    return [
        {"stretch_ratio": float(a), "force_n": float(b)}
        for a, b in zip(stretch, force, strict=True)
    ]


def test_force_quality_tells_a_clean_curve_from_noise():
    stretch = np.linspace(1, 1.17, 72)
    rng = np.random.default_rng(1)

    clean = force_quality(
        _branch(40 * (stretch - 1) ** 1.2 * 6 + rng.normal(0, 0.3, 72))
    )
    assert clean["usable"] is True and clean["snr"] > 10 and clean["spearman"] > 0.95

    noisy = force_quality(_branch(rng.normal(0, 1.0, 72) + 3 * (stretch - 1)))
    assert noisy["usable"] is False and noisy["snr"] < 3

    falling = force_quality(_branch(-30 * (stretch - 1) + rng.normal(0, 0.2, 72)))
    assert (
        falling["usable"] is False and falling["spearman"] < 0
    )  # the wrong sign is no healthy cell

    assert force_quality(_branch([0, 1, 2]))["usable"] is None
    assert force_quality(_branch(np.zeros(20)))["usable"] is False  # nothing to measure


def test_spearman_handles_ties_and_constants():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0


def test_package_records_force_quality_and_tare_states(built):
    pkg = _package(built["untiled"])
    assert pkg["force"]["tare_states"] == 1  # four coarse states: nothing to fit
    assert set(pkg["force"]["quality"]) == {"loading", "release"}
    assert pkg["force"]["quality"]["loading"]["usable"] is None  # fewer than 8 states
    assert (
        built["report"]["orientations"]["warp"]["force"]["quality"]
        == pkg["force"]["quality"]
    )
