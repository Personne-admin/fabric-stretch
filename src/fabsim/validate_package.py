from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image

FORMAT = "fabsim-material-rgb"
PACKAGE_ROOTS = (
    "material_package_v3",
    "material_package_v3_untiled",
    "material_package",
    "material_package_untiled",
)


def _number(value) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _vector(value, length: int, *, positive: bool = False, integer: bool = False) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_number(v) and (not positive or v > 0) and (not integer or isinstance(v, int)) for v in value)
    )


def validate_package(root: Path) -> tuple[list[str], list[str]]:
    root = Path(root)
    if root.name == "package.json":
        root = root.parent
    errors: list[str] = []
    warnings: list[str] = []
    try:
        data = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"package.json: {exc}"], warnings
    if not isinstance(data, dict):
        return ["package.json must be an object"], warnings

    def finite(value, location):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{location}: non-finite number")
        elif isinstance(value, dict):
            for key, item in value.items():
                finite(item, f"{location}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                finite(item, f"{location}[{i}]")

    finite(data, "package")
    if data.get("format") != FORMAT:
        errors.append(f"format must be {FORMAT!r}")
    version = data.get("version")
    if type(version) is not int or version not in (2, 3):
        errors.append("version must be 2 or 3")
    elif version == 2:
        warnings.append("legacy v2: the viewer estimates the force tare; rebuild for v3 layers")

    atlas = data.get("atlas")
    if not isinstance(atlas, dict):
        return errors + ["atlas must be an object"], warnings
    size = atlas.get("dimensions_px")
    if not _vector(size, 2, positive=True, integer=True):
        errors.append("atlas.dimensions_px must be two positive integers")
        size = None
    cells = atlas.get("grid_cells")
    if not _vector(cells, 4, integer=True) or any(v < 0 for v in cells) or any(v == 0 for v in cells[2:]):
        errors.append("atlas.grid_cells must be [col, row, positive cols, positive rows]")
    if atlas.get("origin") != "Loading reference grid top-left":
        errors.append("atlas.origin must be 'Loading reference grid top-left'")
    if atlas.get("active_grid_axis", "vertical") not in ("vertical", "horizontal"):
        errors.append("atlas.active_grid_axis must be vertical or horizontal")

    tiling = data.get("tiling")
    if tiling is not None:
        if not isinstance(tiling, dict) or type(tiling.get("enabled")) is not bool:
            errors.append("tiling must be an object with a boolean enabled field")
        elif tiling["enabled"]:
            tile = tiling.get("tile_size_px")
            period = tiling.get("period_px")
            uv = tiling.get("period_uv")
            valid = (
                size is not None
                and _vector(tile, 2, positive=True, integer=True)
                and _vector(period, 2, positive=True)
                and _vector(uv, 2, positive=True)
                and _vector(tiling.get("origin_uv"), 2)
                and tiling.get("source_atlas_size_px") == size
                and period == tile
                and all(abs(uv[i] - period[i] / size[i]) < 1e-4 for i in range(2))
            )
            if not valid:
                errors.append("inconsistent tiling metadata")
            if _vector(tile, 2, positive=True, integer=True):
                size = tile

    ps = data.get("ps")
    if ps is not None:
        if not isinstance(ps, dict) or not _number(ps.get("albedo_scale")) or ps["albedo_scale"] <= 0:
            errors.append("ps.albedo_scale must be positive and finite")
        if version != 3:
            errors.append("photometric layers require version 3")
    force = data.get("force")
    if force is not None:
        if not isinstance(force, dict):
            errors.append("force must be an object")
        elif force.get("clamp_width_m") is not None:
            width = force["clamp_width_m"]
            if not _number(width) or width <= 0:
                errors.append("force.clamp_width_m must be positive or null")
    residual = data.get("release_residual_stretch")
    if residual is not None and (not _number(residual) or residual <= 0):
        errors.append("release_residual_stretch must be positive or null")

    checked: dict[tuple[Path, int], str | None] = {}
    package_root = root.resolve()

    def image(relative, depth: int, location: str):
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or PureWindowsPath(relative).drive
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            errors.append(f"{location}: path must stay inside the package")
            return
        try:
            path = (package_root / relative).resolve()
            path.relative_to(package_root)
        except (OSError, RuntimeError, ValueError):
            errors.append(f"{location}: path resolves outside the package")
            return
        key = (path, depth)
        if key not in checked:
            issue = None
            try:
                with path.open("rb") as stream:
                    header = stream.read(29)
                if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                    raise ValueError("not a PNG")
                width, height, bits, color = struct.unpack(">IIBB", header[16:26])
                if size is not None and [width, height] != size:
                    raise ValueError(f"size {width}x{height}, expected {size[0]}x{size[1]}")
                if bits != depth or color != 2:
                    raise ValueError(f"expected {depth}-bit RGB PNG")
                with Image.open(path) as img:
                    img.verify()
                # Verification checks chunks; decoding also catches invalid compressed pixel data.
                with Image.open(path) as img:
                    img.load()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                issue = f"{relative}: {exc}"
            checked[key] = issue
        if checked[key]:
            errors.append(f"{location}: {checked[key]}")

    branches = data.get("branches")
    if not isinstance(branches, dict):
        return errors + ["branches must be an object"], warnings
    for branch in ("loading", "release"):
        states = branches.get(branch)
        if not isinstance(states, list) or not states:
            errors.append(f"branches.{branch} must be a non-empty list")
            continue
        stretches = []
        ids = set()
        for i, state in enumerate(states):
            location = f"{branch}[{i}]"
            if not isinstance(state, dict):
                errors.append(f"{location}: state must be an object")
                continue
            stretch = state.get("stretch_ratio")
            if not _number(stretch) or stretch <= 0:
                errors.append(f"{location}.stretch_ratio must be positive and finite")
            else:
                stretches.append(stretch)
            if not _number(state.get("force_n")):
                errors.append(f"{location}.force_n must be finite")
            for field in ("force_raw_n", "travel_mm", "machine_displacement_mm"):
                if state.get(field) is not None and not _number(state[field]):
                    errors.append(f"{location}.{field} must be finite or null")
            frame_id = state.get("frame_id")
            if frame_id is not None:
                if type(frame_id) is not int or frame_id < 0:
                    errors.append(f"{location}.frame_id must be a non-negative integer")
                elif frame_id in ids:
                    errors.append(f"{location}: duplicate frame_id {frame_id}")
                else:
                    ids.add(frame_id)
            image(state.get("texture"), 8, f"{location}.texture")
            for layer in ("albedo", "normal"):
                if ps is not None:
                    image(state.get(layer), 16, f"{location}.{layer}")
                elif state.get(layer) is not None:
                    errors.append(f"{location}.{layer}: layer without a ps block")
        if stretches and max(stretches) - min(stretches) < 1e-6:
            errors.append(f"{branch}: branch has no stretch range")
    return errors, warnings


def package_directories(path: Path) -> list[Path]:
    if path.name == "package.json" or (path / "package.json").is_file():
        return [path.parent if path.name == "package.json" else path]
    roots = [path / name for name in PACKAGE_ROOTS if (path / name).is_dir()]
    return sorted({manifest.parent for root in (roots or [path]) for manifest in root.rglob("package.json")})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fabsim validate-package", description="Check material packages and images."
    )
    parser.add_argument("path", type=Path, help="package.json, package directory, package set, or project")
    args = parser.parse_args(argv)
    packages = package_directories(args.path)
    if not packages:
        print(f"no packages found in {args.path}")
        return 1
    failed = False
    for package in packages:
        errors, warnings = validate_package(package)
        failed |= bool(errors)
        print(f"{package}: {'ERROR' if errors else ('warn' if warnings else 'ok')}")
        for level, messages in (("error", errors), ("warning", warnings)):
            for message in messages:
                print(f"  {level}: {message}")
    return int(failed)
