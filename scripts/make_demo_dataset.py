#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from fabsim import dataset, material_package, ps_frames
from fabsim import photometric as ph

IMAGE_KEYS = ("image_diffuse", *dataset.PS_KEYS)


PACKAGE_DIRS = ("material_package_v3", "material_package_v3_untiled")
DEMO_TILING = [
    "--tiling-window-size",
    "8",
    "16",
    "--tiling-shift",
    "3",
    "--tiling-interp-dist",
    "1",
]
_PATH_LIKE = re.compile(r"^(/|[A-Za-z]:[\\/])")


def pick_frames(frames: list[dict], count: int, keep_ids: set[int]) -> list[dict]:
    index = set(
        np.rint(np.linspace(0, len(frames) - 1, min(count, len(frames))))
        .astype(int)
        .tolist()
    )
    index |= {i for i, f in enumerate(frames) if dataset.frame_id(f, i) in keep_ids}

    return [frames[i] for i in sorted(index)]


def scrub(value):
    if isinstance(value, dict):
        return {
            k: scrub(v)
            for k, v in value.items()
            if not (isinstance(v, str) and _PATH_LIKE.match(v))
        }

    if isinstance(value, list):
        return [scrub(v) for v in value]

    return value


_DROPPED_FRAME_KEYS = ("force_csv_row", "line_load_N_per_m")


def scale_points(frame: dict, sx: float, sy: float) -> dict:
    points = [
        {
            **p,
            "x": round((p["x"] + 0.5) * sx - 0.5, 4),
            "y": round((p["y"] + 0.5) * sy - 0.5, 4),
        }
        for p in frame["points"]
    ]

    return {**frame, "points": points}


def convert_run(
    src: Path, dst: Path, count: int, scale: float, led_config: dict
) -> int:
    frames_doc = json.loads((src / "frames.json").read_text())
    tracks = json.loads((src / "tracks.json").read_text())
    reference = tracks.get("provenance", {}).get("reference_frame")
    frames = pick_frames(
        frames_doc["frames"], count, set() if reference is None else {int(reference)}
    )
    ids = {dataset.frame_id(f, i) for i, f in enumerate(frames)}
    indices = {int(f["force_json_index"]) for f in frames}

    size = None  # (width, height) of the reduced images; the points are scaled by the actual factors
    for frame in frames:
        for key in IMAGE_KEYS:
            source = dataset.image_path(src, frame[key])
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise OSError(f"cannot read {source}")
            height, width = image.shape[:2]
            if size is None:
                size = (width, height)
            elif size != (width, height):
                raise ValueError(
                    f"{source} differs in size from the other images of {src.name}"
                )
            new = (round(width * scale), round(height * scale))
            small = cv2.resize(image, new, interpolation=cv2.INTER_AREA)
            target = dataset.image_path(dst, frame[key])
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), small, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    frames = [
        {k: v for k, v in f.items() if k not in _DROPPED_FRAME_KEYS} for f in frames
    ]
    (dst / "frames.json").write_text(
        json.dumps({**frames_doc, "frames": frames}, indent=1) + "\n"
    )
    forces = json.loads((src / "force.json").read_text())
    kept = [f for f in forces["forces"] if int(f["index"]) in indices]
    (dst / "force.json").write_text(
        json.dumps({**forces, "forces": kept}, indent=1) + "\n"
    )

    session = scrub(json.loads((src / "session.json").read_text()))
    session["count"] = session["frames_written"] = len(frames)
    session["led_config"] = led_config
    if isinstance(session.get("color_calibration"), dict):
        session["color_calibration"].pop("file", None)
    session["note"] = (
        "demo subset: frames and images reduced, see examples/demo/README.md"
    )
    (dst / "session.json").write_text(json.dumps(session, indent=1) + "\n")

    provenance = scrub(tracks["provenance"])
    for key in ("reference_errors_px", "frame_mapping", "inference_dimensions"):
        provenance.pop(key, None)
    new_w, new_h = round(size[0] * scale), round(size[1] * scale)
    sx, sy = new_w / size[0], new_h / size[1]
    provenance["source_dimensions"] = [new_w, new_h]
    provenance["downscaled"] = (
        f"images resized to {new_w}x{new_h}, points scaled by {sx:.5f} x {sy:.5f}"
    )
    tracks_out = {
        **tracks,
        "provenance": provenance,
        "frames": [
            scale_points(f, sx, sy)
            for f in tracks["frames"]
            if int(f["frame_id"]) in ids
        ],
    }
    (dst / "tracks.json").write_text(
        json.dumps(tracks_out, separators=(",", ":")) + "\n"
    )

    return len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a reduced demo dataset from measurement runs.")
    parser.add_argument(
        "project",
        type=Path,
        help="measurement project folder (contains <orientation>_<branch>)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("examples/demo"), help="default: %(default)s"
    )
    parser.add_argument("--orientation", default="warp")
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="frames kept per branch (default: %(default)s)",
    )
    parser.add_argument(
        "--scale", type=float, default=0.15, help="image scale (default: %(default)s)"
    )
    parser.add_argument(
        "--led-config",
        type=Path,
        help="LED geometry of the capture (default: packaged)",
    )
    args = parser.parse_args(argv)

    led_path = args.led_config or ph.default_led_config_path()
    led_config = json.loads(led_path.read_text())
    for name in (*(f"{args.orientation}_{b}" for b in dataset.BRANCHES), *PACKAGE_DIRS):
        shutil.rmtree(
            args.out / name, ignore_errors=True
        )  # keeps the README next to the generated data
    for branch in dataset.BRANCHES:
        name = f"{args.orientation}_{branch}"
        kept = convert_run(
            args.project / name, args.out / name, args.frames, args.scale, led_config
        )
        print(f"{name}: {kept} frames")
    ps_frames.main([str(args.out), "--force"])
    status = material_package.main([str(args.out), *DEMO_TILING])

    return status


if __name__ == "__main__":
    raise SystemExit(main())
