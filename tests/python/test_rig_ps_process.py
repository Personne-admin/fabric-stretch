from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from fabsim import photometric
from fabsim.rig import ps_process, ps_verify
from fabsim.rig.ps_process import PsOptions, RunData

LED = {
    "pol_a_bits": [1, 3, 5],
    "pol_b_bits": [2, 4, 6],
    "diffuse_mask": 1,
    "light_elevation_deg": 45.0,
    "light_azimuths_deg": [91.27, 334.26, 211.16],
}
CFG = photometric.load_light_config(session={"led_config": LED})


def sphere(size=96, albedo=0.5):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    x, y = (xs - size / 2) / (size * 0.4), -(ys - size / 2) / (size * 0.4)

    r2 = x * x + y * y
    n = np.stack([x, y, np.sqrt(np.clip(1 - r2, 0, None))], axis=-1)
    inside = r2 < 0.9
    n[~inside] = (0, 0, 1)

    return (
        np.where(inside, albedo, 0.5).astype(np.float32),
        n.astype(np.float32),
        inside,
    )


def write_run(root, name="run", led=LED, frames=2, gains=(1.0, 1.0, 1.0)):
    run = root / name
    (run / "images").mkdir(parents=True)
    albedo, normals, inside = sphere()
    shots = photometric.render(albedo, normals, CFG, gains)
    diffuse = np.clip(shots.sum(axis=-1) / 2.0, 0, 1)

    records = []
    for i in range(frames):
        record = {
            "frame_id": i,
            "step_index": i,
            "displacement_mm": float(i),
            "is_relaxation_hold": False,
        }
        for k in range(3):
            fname = f"img_{i:04d}_ps_{k + 1}.png"
            cv2.imwrite(
                str(run / "images" / fname),
                np.rint(np.dstack([shots[..., k]] * 3) * 255).astype(np.uint8),
            )
            record[f"image_ps_{k + 1}"] = fname
        fname = f"img_{i:04d}_diffuse.png"
        cv2.imwrite(
            str(run / "images" / fname),
            np.rint(np.dstack([diffuse] * 3) * 255).astype(np.uint8),
        )
        record["image_diffuse"] = fname
        records.append(record)
    (run / "frames.json").write_text(json.dumps({"frames": records}))
    (run / "session.json").write_text(json.dumps({"cam_exposure_ms": 15.0, "led_config": led}))

    return run, normals, inside


def test_detect_mode():
    frames = {
        "frames": [
            {
                "image_ps_1": "a",
                "image_ps_2": "b",
                "image_ps_3": "c",
                "image_ps_b_1": "d",
            }
        ]
    }

    mode, caps = ps_process.detect_mode_from_frames(frames)
    assert mode == "white" and caps["has_polarization"] and caps["ps_count"] == 3
    assert ps_process.detect_mode_from_frames({"frames": [{"image_rgb": "x"}]})[0] == "rgb"
    assert ps_process.detect_mode_from_frames({"frames": []}) == ("unknown", {})


def test_light_comes_from_the_session_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("FABSIM_RIG_CONFIG_DIR", str(tmp_path / "no-user-config"))
    run, _, _ = write_run(tmp_path)
    data = RunData(run)
    assert data.light.source == "session.json:led_config"
    assert data.light.azimuths_deg == (91.27, 334.26, 211.16)


def test_missing_snapshot_falls_back_to_the_explicit_file_then_the_packaged_default(tmp_path, monkeypatch):
    monkeypatch.setenv("FABSIM_RIG_CONFIG_DIR", str(tmp_path / "no-user-config"))
    assert ps_process.resolve_light({}).source == "packaged:led_config.json"
    explicit = tmp_path / "led.json"
    explicit.write_text(json.dumps({"light_azimuths_deg": [0, 120, 240], "light_elevation_deg": 30}))
    assert ps_process.resolve_light({}, explicit).azimuths_deg == (0.0, 120.0, 240.0)
    user = tmp_path / "user-config"
    user.mkdir()
    (user / "led_config.json").write_text(explicit.read_text())
    monkeypatch.setenv("FABSIM_RIG_CONFIG_DIR", str(user))
    assert ps_process.resolve_light({}).source == "file:led_config.json"


def test_snapshot_without_azimuths_raises_instead_of_assuming_0_120_240(tmp_path):
    led = {k: v for k, v in LED.items() if k != "light_azimuths_deg"}
    run, _, _ = write_run(tmp_path, led=led)
    with pytest.raises(ValueError, match="azimuths"):
        RunData(run)


def test_process_frame_recovers_normals_and_albedo(tmp_path):
    run, normals, inside = write_run(tmp_path)
    result = ps_process.process_frame(RunData(run), 1, PsOptions())
    err = np.degrees(np.arccos(np.clip((result.normals * normals).sum(-1), -1, 1)))
    assert np.median(err[inside]) < 1.5
    assert result.albedo_rgb.shape == (96, 96, 3) and result.gains.shape == (3,)
    assert np.allclose(result.gains, 1.0, atol=0.15)
    assert ps_process.albedo_preview_u8(result.albedo_rgb).dtype == np.uint8
    assert ps_process.normals_preview_u8(result.normals, bilateral=True).shape == (
        96,
        96,
        3,
    )


def test_gain_estimation_is_an_option(tmp_path):
    run, _, _ = write_run(tmp_path, gains=(1.2, 1.0, 0.8))
    data = RunData(run)
    est = ps_process.process_frame(data, 0, PsOptions(estimate_gains=True)).gains
    plain = ps_process.process_frame(data, 0, PsOptions(estimate_gains=False)).gains
    assert est[0] > est[2] and np.allclose(plain, 1.0)


def test_unsupported_runs_raise(tmp_path):
    run, _, _ = write_run(tmp_path)
    data = RunData(run)
    data.mode = "rgb"
    with pytest.raises(RuntimeError, match="image_ps"):
        ps_process.process_frame(data, 0, PsOptions())
    with pytest.raises(FileNotFoundError):
        RunData(tmp_path / "nothing")


def test_manifest_records_the_solver_and_light(tmp_path):
    run, _, _ = write_run(tmp_path)
    data = RunData(run)
    manifest = ps_process.manifest(data, PsOptions(), [1, 1, 1], [{"frame_id": 0}])
    assert manifest["solver"] == "fabsim.photometric.solve"
    assert manifest["light"]["azimuths_deg"] == [91.27, 334.26, 211.16]
    json.dumps(manifest)


def test_ps_verify_detail_report(tmp_path, capsys):
    run, _, _ = write_run(tmp_path)
    code = ps_verify.main([str(run), "--frame", "1"])
    summary = (run / "ps_verify" / "summary.txt").read_text()
    assert code in (0, 1)
    assert "CHECK 1" in summary and "re-rendering" in summary and "OVERALL VERDICT" in summary
    assert (run / "ps_verify" / "frame0001_rerender_compare.png").is_file()
    assert "source: session.json:led_config" in summary
    capsys.readouterr()


def test_ps_verify_comparison_table(tmp_path, capsys):
    write_run(tmp_path / "tests", "a", frames=1)
    write_run(tmp_path / "tests", "b", frames=1)
    code = ps_verify.main([str(tmp_path / "tests"), "--sample", "1"])
    out = capsys.readouterr().out
    assert code in (0, 1) and "ps_compare.txt" in out
    assert "PS CAPTURE-CONFIG COMPARISON (2 runs" in (tmp_path / "tests" / "ps_compare.txt").read_text()


@pytest.mark.parametrize("module", ["fabsim.rig.ps_verify", "fabsim.rig.ps_process", "fabsim.rig.dispatcher"])
def test_headless_modules_do_not_import_tkinter(module):
    code = f"import sys, {module}; sys.exit(1 if 'tkinter' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
