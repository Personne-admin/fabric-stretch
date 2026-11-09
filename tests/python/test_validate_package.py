from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from fabsim import cli
from fabsim.validate_package import main, validate_package


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    Image.new("RGB", (8, 6)).save(root / "rgb.png")
    cv2.imwrite(str(root / "map.png"), np.zeros((6, 8, 3), dtype=np.uint16))
    state = {
        "stretch_ratio": 1.0,
        "force_n": 0.0,
        "texture": "rgb.png",
        "albedo": "map.png",
        "normal": "map.png",
    }
    data = {
        "format": "fabsim-material-rgb",
        "version": 3,
        "atlas": {
            "dimensions_px": [8, 6],
            "grid_cells": [0, 0, 2, 2],
            "origin": "Loading reference grid top-left",
        },
        "ps": {"albedo_scale": 1.5},
        "branches": {
            branch: [dict(state), {**state, "stretch_ratio": 1.2}] for branch in ("loading", "release")
        },
    }
    save(root, data)
    return root, data


def save(root, data):
    (root / "package.json").write_text(json.dumps(data))


def test_demo_and_cli_discovery(capsys):
    demo = Path(__file__).resolve().parents[2] / "examples/demo"
    assert cli.main(["validate-package", str(demo)]) == 0
    assert capsys.readouterr().out.count(": ok") == 2


def test_valid_layers_and_single_manifest_input(package):
    root, _ = package
    assert validate_package(root) == ([], [])
    assert main([str(root / "package.json")]) == 0


@pytest.mark.parametrize("path", ["../outside.png", "/tmp/outside.png", r"C:\outside.png", "", None])
def test_paths_cannot_leave_the_package(package, path):
    root, data = package
    data["branches"]["loading"][0]["texture"] = path
    save(root, data)
    errors, _ = validate_package(root)
    assert any("path must stay inside" in error for error in errors)


def test_symlink_escape_is_rejected(package):
    root, _ = package
    outside = root.parent / "outside.png"
    shutil.copyfile(root / "rgb.png", outside)
    (root / "rgb.png").unlink()
    (root / "rgb.png").symlink_to(outside)
    assert any("resolves outside" in error for error in validate_package(root)[0])


@pytest.mark.parametrize("problem", ["missing", "dimensions", "depth", "corrupt"])
def test_bad_image_files(package, problem):
    root, _ = package
    if problem == "missing":
        (root / "map.png").unlink()
    elif problem == "dimensions":
        Image.new("RGB", (2, 3)).save(root / "rgb.png")
    elif problem == "depth":
        Image.new("RGB", (8, 6)).save(root / "map.png")
    else:
        path = root / "rgb.png"
        content = bytearray(path.read_bytes())
        content[45] ^= 0xFF
        path.write_bytes(content)
    assert validate_package(root)[0]
    assert main([str(root)]) == 1


@pytest.mark.parametrize(
    "problem", ["nan", "empty", "range", "incomplete", "orphan", "tiling", "version", "duplicate"]
)
def test_bad_manifest_data(package, problem):
    root, data = package
    loading = data["branches"]["loading"]
    if problem == "nan":
        loading[0]["force_n"] = float("nan")
    elif problem == "empty":
        data["branches"]["release"] = []
    elif problem == "range":
        loading[1]["stretch_ratio"] = 1
    elif problem == "incomplete":
        del loading[0]["normal"]
    elif problem == "orphan":
        del data["ps"]
    elif problem == "tiling":
        data["tiling"] = {"enabled": True}
    elif problem == "version":
        data["version"] = 99
    else:
        loading[0]["frame_id"] = loading[1]["frame_id"] = 2
    save(root, data)
    assert validate_package(root)[0]


def test_v2_colour_package_is_valid_with_warning(package):
    root, data = package
    data["version"] = 2
    del data["ps"]
    for states in data["branches"].values():
        for state in states:
            del state["albedo"], state["normal"]
    save(root, data)
    errors, warnings = validate_package(root)
    assert not errors and len(warnings) == 1


def test_tiled_dimensions_and_uv_consistency(package):
    root, data = package
    data["atlas"]["dimensions_px"] = [16, 12]
    data["tiling"] = {
        "enabled": True,
        "tile_size_px": [8, 6],
        "source_atlas_size_px": [16, 12],
        "period_px": [8, 6],
        "origin_uv": [0.1, 0.2],
        "period_uv": [0.5, 0.5],
    }
    save(root, data)
    assert not validate_package(root)[0]
    data["tiling"]["period_uv"] = [1.0, 1.0]
    save(root, data)
    assert "inconsistent tiling metadata" in validate_package(root)[0]


def test_no_packages_and_malformed_json(tmp_path):
    assert main([str(tmp_path)]) == 1
    (tmp_path / "package.json").write_text("{")
    assert main([str(tmp_path)]) == 1
    (tmp_path / "package.json").write_text("[]")
    assert validate_package(tmp_path)[0] == ["package.json must be an object"]
