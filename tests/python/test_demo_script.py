from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import pytest

import synthetic_project as syn
from fabsim import dataset

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_demo_dataset.py"
pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="scripts/ is not present")


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("make_demo_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_tracks_follow_the_actual_resize_factor_and_stale_fields_are_dropped(script, tmp_path):
    src = tmp_path / "src" / "warp_loading"
    syn.write_run(src, "loading", count=4)
    frames_doc = json.loads((src / "frames.json").read_text())
    for frame in frames_doc["frames"]:
        frame["line_load_N_per_m"] = -21.3  # derived from the untared force
        frame["force_csv_row"] = 7

    (src / "frames.json").write_text(json.dumps(frames_doc))
    scale = 0.1234  # 320 px -> 39.49 -> 39 px, a different factor than the nominal one
    dst = tmp_path / "dst" / "warp_loading"
    assert script.convert_run(src, dst, 3, scale, {"light_azimuths_deg": [0, 120, 240]}) == 3

    first = dataset.load_frames(dst)[0]
    image = cv2.imread(str(dataset.image_path(dst, first["image_diffuse"])))
    assert image.shape[:2] == (39, 39)
    assert not {"line_load_N_per_m", "force_csv_row"} & set(first)

    factor = 39 / syn.SIZE
    source = json.loads((src / "tracks.json").read_text())["frames"][0]["points"][1]
    kept = json.loads((dst / "tracks.json").read_text())
    point = next(p for p in kept["frames"][0]["points"] if p["id"] == source["id"])
    assert point["x"] == pytest.approx((source["x"] + 0.5) * factor - 0.5, abs=1e-3)
    assert point["y"] == pytest.approx((source["y"] + 0.5) * factor - 0.5, abs=1e-3)
    assert kept["provenance"]["source_dimensions"] == [39, 39]
    assert json.loads((dst / "session.json").read_text())["led_config"]["light_azimuths_deg"] == [0, 120, 240]
