from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from fabsim import dataset, material_package
from fabsim import photometric as ph

DEMO = Path(__file__).resolve().parents[2] / "examples" / "demo"
pytestmark = pytest.mark.skipif(
    not DEMO.is_dir(), reason="examples/demo is not present"
)


def _package(directory):
    return json.loads((directory / "warp" / "package.json").read_text())


def test_sources_are_consistent():
    runs = dataset.discover_runs(DEMO)
    assert [r.name for r in runs] == ["warp_loading", "warp_release"]
    for run in runs:
        frames = dataset.load_frames(run.path)
        forces = dataset.load_force(run.path)
        session = dataset.load_session(run.path)

        assert session["count"] == len(frames) >= 8
        assert ph.load_light_config(session=session).source == "session.json:led_config"

        tracks, _, cols, rows = material_package.grid_frames(run.path / "tracks.json")
        for position, frame in enumerate(frames):
            assert dataset.frame_id(frame, position) in tracks
            assert dataset.force_for_frame(frame, forces) is not None
            assert dataset.has_ps(frame)
            for key in ("image_diffuse", *dataset.PS_KEYS):
                assert dataset.image_path(run.path, frame[key]).is_file()

        manifest = json.loads((run.path / "ps" / "ps_manifest.json").read_text())
        assert len(manifest["frames"]) == len(frames)


@pytest.mark.parametrize("name", ["material_package_v3", "material_package_v3_untiled"])
def test_committed_packages_are_v3_with_photometric_layers(name):
    pkg = _package(DEMO / name)
    assert (pkg["format"], pkg["version"]) == ("fabsim-material-rgb", 3)
    assert pkg["release_residual_stretch"] >= 1
    assert pkg["force"]["clamp_width_m"] == pytest.approx(0.202)
    assert ("tiling" in pkg) == (name == "material_package_v3")

    width, height = (
        pkg["tiling"]["tile_size_px"]
        if "tiling" in pkg
        else pkg["atlas"]["dimensions_px"]
    )

    for branch, states in pkg["branches"].items():
        assert len(states) == 8
        for state in states:
            for key in ("texture", "albedo", "normal"):
                image = cv2.imread(
                    str(DEMO / name / "warp" / state[key]), cv2.IMREAD_UNCHANGED
                )
                assert image.shape == (height, width, 3), (branch, key)

        assert states[0]["force_n"] == 0 or branch == "release"


def test_rebuild_matches_the_committed_untiled_package(tmp_path):
    report = material_package.build(DEMO, None, tmp_path)
    assert report["orientations"]["warp"]["status"] == "built"

    mine, committed = _package(tmp_path), _package(DEMO / "material_package_v3_untiled")
    assert mine["release_residual_stretch"] == pytest.approx(
        committed["release_residual_stretch"], abs=1e-3
    )

    for branch in ("loading", "release"):
        for a, b in zip(
            mine["branches"][branch], committed["branches"][branch], strict=True
        ):
            assert a["frame_id"] == b["frame_id"]
            assert a["stretch_ratio"] == pytest.approx(b["stretch_ratio"], abs=1e-3)
            assert a["force_n"] == pytest.approx(b["force_n"], abs=1e-6)
