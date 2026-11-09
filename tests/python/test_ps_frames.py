from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fabsim import dataset, ps_frames
from fabsim import photometric as ph
from fabsim.cli import main as cli_main

CFG = ph.load_light_config()


def synthetic_scene(size=48):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)

    tilt = 0.25 * np.stack([np.sin(xs / 6), np.cos(ys / 7)], axis=-1)
    n = np.dstack([tilt, np.ones((size, size), np.float32)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    albedo = 0.3 + 0.4 * (xs / size)

    return albedo.astype(np.float32), n


def write_run(path, gains, n_frames=2, with_ps=True):
    (path / "images").mkdir(parents=True)
    albedo, n = synthetic_scene()

    frames = []
    for i in range(n_frames):
        frame = {"frame_id": i, "image_diffuse": f"img_{i:04d}_diffuse.png"}
        diffuse = np.stack([albedo * 1.2, albedo, albedo * 0.6], axis=-1).clip(0, 1)
        cv2.imwrite(
            str(path / "images" / frame["image_diffuse"]),
            cv2.cvtColor(np.rint(diffuse * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )

        if with_ps:
            shots = ph.render(albedo, n, CFG, gains)
            for k, key in enumerate(dataset.PS_KEYS):
                frame[key] = f"img_{i:04d}_ps_{k + 1}.png"
                gray = np.rint(shots[..., k].clip(0, 1) * 255).astype(np.uint8)
                cv2.imwrite(str(path / "images" / frame[key]), gray)

        frames.append(frame)

    (path / "frames.json").write_text(json.dumps({"frames": frames}))

    return albedo, n


def test_ps_frames_end_to_end(tmp_path):
    gains = np.array([1.2, 0.7, 1.1])
    gains /= gains.mean()
    albedo, n = write_run(tmp_path / "warp_loading", gains)
    write_run(tmp_path / "warp_release", gains)
    write_run(tmp_path / "weft_loading", gains, with_ps=False)

    status = ps_frames.run_project(tmp_path, jobs=2)
    assert status["warp_loading"] == "done: 2 frames"
    assert status["warp_release"] == "done: 2 frames"
    assert status["weft_loading"].startswith("skipped")
    assert not (tmp_path / "weft_loading" / "ps").exists()

    out = tmp_path / "warp_loading" / "ps"
    manifest = json.loads((out / "ps_manifest.json").read_text())
    assert manifest["format"] == "fabsim-ps-frames" and manifest["version"] == 1
    assert manifest["light"]["source"].startswith("packaged")
    np.testing.assert_allclose(manifest["gains"], gains, atol=0.06)
    assert [f["frame_id"] for f in manifest["frames"]] == [0, 1]

    # the release run shares the loading reference measurement
    release = json.loads(
        (tmp_path / "warp_release" / "ps" / "ps_manifest.json").read_text()
    )
    assert (
        release["albedo_scale"] == manifest["albedo_scale"]
        and release["gains"] == manifest["gains"]
    )

    normal = cv2.cvtColor(
        cv2.imread(str(out / manifest["frames"][0]["normal"]), cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGR2RGB,
    )
    assert normal.dtype == np.uint16
    assert (
        np.degrees(
            np.arccos(np.clip((ph.decode_normals_u16(normal) * n).sum(-1), -1, 1))
        ).mean()
        < 5.0
    )
    albedo_png = cv2.imread(
        str(out / manifest["frames"][0]["albedo"]), cv2.IMREAD_UNCHANGED
    )
    assert albedo_png.dtype == np.uint8 and albedo_png.shape[2] == 3
    gray = (
        cv2.cvtColor(albedo_png, cv2.COLOR_BGR2RGB)
        @ ph.LUMA
        / 255
        * manifest["albedo_scale"]
    )
    assert np.abs(gray - albedo).mean() < 0.02

    # a second run without --force leaves existing output alone
    assert ps_frames.run_project(tmp_path)["warp_loading"].startswith(
        "skipped: ps/ exists"
    )
    assert (
        ps_frames.run_project(tmp_path, runs=["warp_loading"], force=True)[
            "warp_loading"
        ]
        == "done: 2 frames"
    )


def test_cli_dispatch(tmp_path, capsys):
    write_run(tmp_path / "warp_loading", np.ones(3))
    assert cli_main(["ps", str(tmp_path), "--jobs", "1"]) == 0
    assert "warp_loading: done" in capsys.readouterr().out
    assert cli_main(["--version"]) == 0
    assert cli_main(["nonsense"]) == 2
    assert cli_main([]) == 2
    assert cli_main(["--help"]) == 0
    assert "package" in capsys.readouterr().out


def test_led_config_file_is_the_fallback_for_runs_without_a_snapshot(tmp_path):
    led = tmp_path / "other_led.json"
    led.write_text(
        json.dumps({"light_azimuths_deg": [0, 120, 240], "light_elevation_deg": 30})
    )
    project = tmp_path / "project"
    write_run(project / "warp_loading", np.ones(3), n_frames=1)
    write_run(project / "weft_loading", np.ones(3), n_frames=1)
    snapshot = {"light_azimuths_deg": [10, 130, 250], "light_elevation_deg": 40}
    session = project / "warp_loading" / "session.json"
    session.write_text(json.dumps({"led_config": snapshot}))
    ps_frames.run_project(project, led_config=led)
    sources = {
        run: json.loads((project / run / "ps" / "ps_manifest.json").read_text())[
            "light"
        ]
        for run in ("warp_loading", "weft_loading")
    }
    assert (
        sources["warp_loading"]["source"] == "session.json:led_config"
    )  # the snapshot wins
    assert sources["warp_loading"]["azimuths_deg"] == [10, 130, 250]
    assert sources["weft_loading"]["source"] == "file:other_led.json"


def test_demo_products_rerender_the_captured_photos():
    run = Path(__file__).resolve().parents[2] / "examples" / "demo" / "warp_loading"
    if not run.is_dir():
        pytest.skip("examples/demo is not present")
    manifest = json.loads((run / "ps" / "ps_manifest.json").read_text())
    cfg = ph.LightConfig(
        tuple(manifest["light"]["azimuths_deg"]),
        manifest["light"]["elevation_deg"],
        tuple(manifest["gains"]),
    )
    directions = ph.light_matrix(cfg)
    for entry in manifest["frames"]:
        cfg = ph.LightConfig(
            cfg.azimuths_deg,
            cfg.elevation_deg,
            tuple(entry.get("gains", manifest["gains"])),
        )
        albedo = cv2.cvtColor(
            cv2.imread(str(run / "ps" / entry["albedo"])), cv2.COLOR_BGR2RGB
        )
        albedo = albedo.astype(np.float32) / 255 * manifest["albedo_scale"]
        normal = cv2.cvtColor(
            cv2.imread(str(run / "ps" / entry["normal"]), cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB,
        )
        n = ph.decode_normals_u16(normal)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        frame = next(
            f
            for f in json.loads((run / "frames.json").read_text())["frames"]
            if f["frame_id"] == entry["frame_id"]
        )
        photos = ph.read_ps_gray(
            [run / "images" / frame[f"image_ps_{i}"] for i in (1, 2, 3)]
        )
        for i in range(3):
            shaded = ph.shade(albedo, n, directions[i], gain=cfg.gains[i])
            assert np.abs(shaded @ ph.LUMA - photos[..., i]).mean() < 1 / 255


def _rewrite_ps(run, frame_index, gains):
    albedo, n = synthetic_scene()
    shots = ph.render(albedo, n, CFG, gains)
    for k in range(3):
        gray = np.rint(shots[..., k].clip(0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(run / "images" / f"img_{frame_index:04d}_ps_{k + 1}.png"), gray)


def test_gains_follow_the_frames_unless_the_configuration_fixes_them(tmp_path):
    base = np.array([1.2, 0.7, 1.1])
    base /= base.mean()
    drifted = base * [1.0, 0.75, 1.0]
    drifted /= drifted.mean()
    write_run(tmp_path / "warp_loading", base)
    _rewrite_ps(tmp_path / "warp_loading", 1, drifted)
    ps_frames.run_project(tmp_path, jobs=1)
    manifest = json.loads(
        (tmp_path / "warp_loading" / "ps" / "ps_manifest.json").read_text()
    )

    assert manifest["gains_mode"] == "per-frame"
    np.testing.assert_allclose(
        manifest["gains"], base, atol=0.06
    )  # the reference frame

    first, second = (e["gains"] for e in manifest["frames"])
    np.testing.assert_allclose(first, base, atol=0.06)
    np.testing.assert_allclose(second, drifted, atol=0.06)

    # the normals of the drifted frame stay flat although its gains differ from the reference
    normal = cv2.cvtColor(
        cv2.imread(
            str(tmp_path / "warp_loading" / "ps" / manifest["frames"][1]["normal"]), -1
        ),
        cv2.COLOR_BGR2RGB,
    )

    _, n = synthetic_scene()
    angle = np.degrees(
        np.arccos(np.clip((ph.decode_normals_u16(normal) * n).sum(-1), -1, 1))
    )

    assert angle.mean() < 5.0

    fixed = tmp_path / "fixed"
    write_run(fixed / "warp_loading", base)
    _rewrite_ps(fixed / "warp_loading", 1, drifted)
    (fixed / "warp_loading" / "session.json").write_text(
        json.dumps(
            {
                "led_config": {
                    "light_azimuths_deg": list(CFG.azimuths_deg),
                    "light_elevation_deg": CFG.elevation_deg,
                    "gains": [1.0, 1.0, 1.0],
                }
            }
        )
    )

    ps_frames.run_project(fixed, jobs=1)
    manifest = json.loads(
        (fixed / "warp_loading" / "ps" / "ps_manifest.json").read_text()
    )

    assert manifest["gains_mode"] == "fixed"
    assert [e["gains"] for e in manifest["frames"]] == [[1.0, 1.0, 1.0]] * 2


def test_an_unreadable_frame_is_skipped_and_listed(tmp_path, capsys):
    write_run(tmp_path / "warp_loading", np.ones(3), n_frames=3)
    broken = tmp_path / "warp_loading" / "images" / "img_0001_ps_2.png"
    broken.write_bytes(broken.read_bytes()[:-30])  # cut short, as a damaged copy is
    status = ps_frames.run_project(tmp_path, jobs=1)

    assert status["warp_loading"] == "done: 2 frames, 1 unreadable frame(s) skipped"
    assert "frame 1 skipped" in capsys.readouterr().err
    manifest = json.loads(
        (tmp_path / "warp_loading" / "ps" / "ps_manifest.json").read_text()
    )

    assert [f["frame_id"] for f in manifest["frames"]] == [0, 2]
    assert [f["frame_id"] for f in manifest["skipped_frames"]] == [1]
    assert "cannot read" in manifest["skipped_frames"][0]["reason"]


def test_quality_is_recorded_and_a_poor_correlation_warns(tmp_path):
    write_run(tmp_path / "good" / "warp_loading", np.ones(3), n_frames=1)
    status = ps_frames.run_project(tmp_path / "good", jobs=1)

    assert status["warp_loading"] == "done: 1 frames"
    quality = json.loads(
        (tmp_path / "good" / "warp_loading" / "ps" / "ps_manifest.json").read_text()
    )["quality"]

    assert quality["albedo_diffuse_correlation"] > 0.95
    assert quality["exposure_gain"] > 0 and quality["warnings"] == []
    assert quality["mean_normal_tilt_deg"] < 5

    # a negative diffuse photograph: the capture and the diffuse shot disagree
    write_run(tmp_path / "bad" / "warp_loading", np.ones(3), n_frames=1)
    image = tmp_path / "bad" / "warp_loading" / "images" / "img_0000_diffuse.png"
    cv2.imwrite(str(image), 255 - cv2.imread(str(image)))
    status = ps_frames.run_project(tmp_path / "bad", jobs=1)
    assert (
        "warning: warp_loading: albedo and diffuse photograph correlate poorly"
        in status["warp_loading"]
    )
    manifest = json.loads(
        (tmp_path / "bad" / "warp_loading" / "ps" / "ps_manifest.json").read_text()
    )
    assert manifest["quality"]["albedo_diffuse_correlation"] < 0
    assert manifest["quality"]["warnings"]
