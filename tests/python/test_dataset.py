from __future__ import annotations

import json

import numpy as np
from PIL import Image

from fabsim import dataset
from fabsim.validate import main as validate_main
from fabsim.validate import validate_run


def make_run(path, n_frames=3, ps=True, force_status="ok"):
    (path / "images").mkdir(parents=True)
    frames, forces = [], []
    for i in range(n_frames):
        frame = {
            "frame_id": i,
            "force_json_index": i,
            "image_diffuse": f"img_{i:04d}_diffuse.png",
        }

        Image.fromarray(np.full((8, 8, 3), 100, np.uint8)).save(path / "images" / frame["image_diffuse"])

        if ps:
            for k, key in enumerate(dataset.PS_KEYS, 1):
                frame[key] = f"img_{i:04d}_ps_{k}.png"
                Image.fromarray(np.full((8, 8, 3), 50 * k, np.uint8)).save(path / "images" / frame[key])

        frames.append(frame)
        forces.append({"index": i, "force_n": float(i), "status": force_status})

    (path / "frames.json").write_text(json.dumps({"frames": frames}))
    (path / "force.json").write_text(json.dumps({"forces": forces}))
    (path / "session.json").write_text("{}")
    (path / "tracks.json").write_text("{}")


def test_discover_runs_sorted_and_filtered(tmp_path):
    for name in ("weft_release", "warp_release", "warp_loading", "warp_bias_loading"):
        make_run(tmp_path / name, n_frames=1)

    (tmp_path / "material_package").mkdir()
    (tmp_path / "notes_loading").mkdir()  # no frames.json: not a run

    runs = dataset.discover_runs(tmp_path)
    assert [r.name for r in runs] == [
        "warp_loading",
        "warp_release",
        "warp_bias_loading",
        "weft_release",
    ]

    assert runs[2].orientation == "warp_bias"
    assert [r.name for r in dataset.discover_runs(tmp_path, ["weft"])] == ["weft_release"]

    grouped = dataset.discover_orientations(tmp_path)
    assert set(grouped["warp"]) == {"loading", "release"} and set(grouped["weft"]) == {"release"}

    assert dataset.discover_runs(tmp_path / "missing") == []


def test_force_is_matched_by_index_and_status(tmp_path):
    run = tmp_path / "warp_loading"
    make_run(run, n_frames=3)
    forces = [
        {"index": 0, "force_n": 1.0, "status": "ok"},
        {"index": 1, "force_n": 2.0, "status": "timeout"},
        {"index": 5, "force_n": 3.0, "status": "ok"},
        {"index": 2, "force_n": float("nan"), "status": "ok"},
    ]

    (run / "force.json").write_text(json.dumps({"forces": forces}))

    table = dataset.load_force(run)
    assert set(table) == {0, 5}

    frames = dataset.load_frames(run)
    assert dataset.force_for_frame(frames[0], table)["force_n"] == 1.0
    assert dataset.force_for_frame(frames[1], table) is None
    assert dataset.force_for_frame({"force_json_index": 5}, table)["force_n"] == 3.0
    assert dataset.force_for_frame({}, table) is None


def test_helpers(tmp_path):
    run = tmp_path / "warp_loading"
    make_run(run, n_frames=1)

    frame = dataset.load_frames(run)[0]
    assert dataset.has_ps(frame) and not dataset.has_ps({"image_ps_1": "a.png"})
    assert dataset.image_path(run, "a.png") == run / "images" / "a.png"
    assert dataset.frame_id({}, 7) == 7 and dataset.frame_id({"frame_id": 3}, 7) == 3
    assert dataset.load_session(tmp_path / "nothing") == {}
    assert dataset.load_force(tmp_path / "nothing") == {}


def test_validate_run_reports_problems(tmp_path):
    good = tmp_path / "warp_loading"
    make_run(good, ps=False)
    errors, warnings = validate_run(good)
    assert errors == [] and any("photometric" in w for w in warnings)

    bad = tmp_path / "warp_release"
    make_run(bad, force_status="error")
    (bad / "images" / "img_0000_diffuse.png").unlink()
    errors, _ = validate_run(bad)
    assert any("missing images" in e for e in errors) and any("force" in e for e in errors)

    assert validate_main([str(tmp_path)]) == 1
    assert validate_main([str(tmp_path / "empty")]) == 1


def test_validate_lists_folders_that_look_like_runs_but_are_ignored(tmp_path, capsys):
    make_run(tmp_path / "warp_loading", n_frames=1)
    make_run(tmp_path / "warp_loading_2", n_frames=1)
    (tmp_path / "material_package").mkdir()
    assert dataset.ignored_run_folders(tmp_path) == [
        (
            "warp_loading_2",
            "name is not <orientation>_<loading|release>; rename it to replace warp_loading",
        )
    ]
    assert validate_main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "warp_loading_2: warn" in out and "ignored by the pipeline" in out
    assert dataset.warn_ignored_run_folders(tmp_path)
    assert "warp_loading_2" in capsys.readouterr().err


def test_image_problem_detects_missing_and_cut_short_pngs(tmp_path):
    good = tmp_path / "good.png"
    Image.fromarray(np.full((64, 64, 3), 90, np.uint8)).save(good)
    assert dataset.image_problem(good) is None
    assert "missing" in dataset.image_problem(tmp_path / "absent.png")
    cut = tmp_path / "cut.png"
    cut.write_bytes(good.read_bytes()[:-20])  # a copy that stopped early
    assert "truncated" in dataset.image_problem(cut)
    tiny = tmp_path / "tiny.png"
    tiny.write_bytes(b"\x89PNG")
    assert "truncated" in dataset.image_problem(tiny)
