from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import dataset
from . import photometric as ph

MANIFEST = "ps_manifest.json"
ALBEDO_ENCODING = "linear RGB 8-bit; albedo = value/255*albedo_scale"
NORMAL_ENCODING = (
    "16-bit RGB; n = 2*v/65535-1; x=image right, y=image up, z=toward camera"
)


def _frame_inputs(run: Path, frame: dict):
    ps = ph.read_ps_gray(
        [dataset.image_path(run, frame[key]) for key in dataset.PS_KEYS]
    )

    diffuse = ph.read_rgb_linear(dataset.image_path(run, frame["image_diffuse"]))
    if diffuse.shape[:2] != ps.shape[:2]:
        raise ValueError("diffuse and PS images differ in size")

    return ps, diffuse


def reference_frame(run: Path, frames: list[dict]) -> dict | None:
    with_ps = [f for f in frames if dataset.has_ps(f) and f.get("image_diffuse")]
    if not with_ps:
        return None

    tracks = run / "tracks.json"
    if tracks.is_file():
        wanted = (
            json.loads(tracks.read_text()).get("provenance", {}).get("reference_frame")
        )

        for frame in with_ps:
            if dataset.frame_id(frame, frames.index(frame)) == wanted:
                return frame

    return with_ps[0]


@dataclass(frozen=True)
class Reference:
    light: ph.LightConfig  # gains of the reference frame filled in
    albedo_scale: float
    exposure_gain: float  # all-LED diffuse brightness over single-LED albedo brightness
    quality: dict
    track_gains: (
        bool  # True: gains are estimated again for every frame (none were configured)
    )
    warnings: list[str] = field(default_factory=list)


def frame_light(
    reference: Reference, ps: np.ndarray, support: np.ndarray | None = None
) -> ph.LightConfig:
    if not reference.track_gains:
        return reference.light

    if support is None:
        support = ph.support_mask(ps)

    return reference.light.with_gains(ph.estimate_gains(ps, support))


def measure_reference(run: Path, frames: list[dict], cfg: ph.LightConfig) -> Reference:
    frame = reference_frame(run, frames)
    if frame is None:
        raise ValueError(f"{run.name}: no frame with PS images")

    ps, diffuse = _frame_inputs(run, frame)
    support = ph.support_mask(ps)

    track = cfg.gains is None
    if track:
        cfg = cfg.with_gains(ph.estimate_gains(ps, support))

    albedo, normals = ph.solve(ps, cfg)
    colour = ph.color_albedo(albedo, diffuse)
    quality = {
        "reference_frame_id": dataset.frame_id(frame, frames.index(frame)),
        "albedo_diffuse_correlation": round(
            ph.albedo_correlation(albedo, diffuse, support), 3
        ),
        "mean_normal_tilt_deg": round(ph.mean_normal_tilt_deg(normals, support), 2),
        "exposure_gain": ph.exposure_gain(albedo, normals, diffuse, support),
    }

    warnings = quality_warnings(run.name, quality)

    return Reference(
        cfg,
        ph.albedo_scale(colour, support),
        quality["exposure_gain"],
        quality,
        track,
        warnings,
    )


def quality_warnings(name: str, quality: dict) -> list[str]:
    r = quality["albedo_diffuse_correlation"]
    if r < ph.MIN_ALBEDO_CORRELATION:
        return [
            f"{name}: albedo and diffuse photograph correlate poorly (r = {r:.2f}, expected above "
            f"{ph.MIN_ALBEDO_CORRELATION}), check the photometric-stereo capture."
        ]

    return []


def _process_frame(task):
    run, frame, fid, reference, out = task
    try:
        ps, diffuse = _frame_inputs(Path(run), frame)
        cfg = frame_light(reference, ps)
        albedo, normals = ph.solve(ps, cfg)

        rgb = ph.encode_albedo_u8(
            ph.color_albedo(albedo, diffuse), reference.albedo_scale
        )

        cv2.imwrite(
            str(Path(out) / f"img_{fid:04d}_albedo.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )

        normal = cv2.cvtColor(ph.encode_normals_u16(normals), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(out) / f"img_{fid:04d}_normal.png"), normal)
    except (OSError, ValueError) as exc:
        return fid, str(exc), None

    return fid, None, [round(float(g), 5) for g in cfg.gains]


def process_run(run: Path, reference: Reference, jobs: int = 1) -> dict:
    frames = dataset.load_frames(run)
    out = run / "ps"
    out.mkdir(exist_ok=True)
    tasks = [
        (str(run), frame, dataset.frame_id(frame, position), reference, str(out))
        for position, frame in enumerate(frames)
        if dataset.has_ps(frame) and frame.get("image_diffuse")
    ]

    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_process_frame, tasks, chunksize=2))
    else:
        results = [_process_frame(task) for task in tasks]

    failed = {fid: error for fid, error, _ in results if error}
    for fid, error in sorted(failed.items()):
        print(f"  warning: {run.name} frame {fid} skipped: {error}", file=sys.stderr)

    entries = [
        {
            "frame_id": fid,
            "albedo": f"img_{fid:04d}_albedo.png",
            "normal": f"img_{fid:04d}_normal.png",
            "gains": gains,
        }
        for fid, _, gains in sorted(results, key=lambda r: r[0])
        if fid not in failed
    ]

    cfg = reference.light
    manifest = {
        "format": "fabsim-ps-frames",
        "version": 1,
        "light": {
            "azimuths_deg": list(cfg.azimuths_deg),
            "elevation_deg": cfg.elevation_deg,
            "source": cfg.source,
        },
        "gains": list(cfg.gains),
        "gains_mode": "per-frame" if reference.track_gains else "fixed",
        "albedo_scale": reference.albedo_scale,
        "albedo_encoding": ALBEDO_ENCODING,
        "normal_encoding": NORMAL_ENCODING,
        "quality": {**reference.quality, "warnings": reference.warnings},
        "frames": entries,
    }
    if failed:
        manifest["skipped_frames"] = [
            {"frame_id": fid, "reason": failed[fid]} for fid in sorted(failed)
        ]

    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest


def run_project(
    project: Path,
    runs: list[str] | None = None,
    led_config: Path | None = None,
    jobs: int = 1,
    force: bool = False,
) -> dict[str, str]:
    discovered = dataset.discover_runs(project)
    if runs:
        unknown = set(runs) - {r.name for r in discovered}
        if unknown:
            raise SystemExit(f"unknown run(s): {', '.join(sorted(unknown))}")

    status: dict[str, str] = {}
    shared: dict[str, Reference] = {}  # per orientation, from its loading run
    for run in sorted(discovered, key=lambda r: (r.orientation, r.branch != "loading")):
        frames = dataset.load_frames(run.path)
        if reference_frame(run.path, frames) is None:
            status[run.name] = "skipped: no PS images"
            continue

        selected = not runs or run.name in runs
        cfg = ph.load_light_config(led_config, session=dataset.load_session(run.path))
        if run.orientation not in shared:
            shared[run.orientation] = measure_reference(run.path, frames, cfg)

        reference = shared[run.orientation]
        if not selected:
            continue

        if (run.path / "ps" / MANIFEST).is_file() and not force:
            status[run.name] = "skipped: ps/ exists (use --force)"
            continue

        manifest = process_run(run.path, reference, jobs)
        text = f"done: {len(manifest['frames'])} frames"
        if "skipped_frames" in manifest:
            text += f", {len(manifest['skipped_frames'])} unreadable frame(s) skipped"

        status[run.name] = "; ".join(
            [text, *(f"warning: {w}" for w in reference.warnings)]
        )

    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fabsim ps",
        description="Process photometric-stereo images for measurement runs.",
    )
    parser.add_argument(
        "project",
        type=Path,
        help="project folder containing <orientation>_<branch> runs",
    )
    parser.add_argument(
        "--runs", nargs="+", metavar="RUN", help="only these run folders (default: all)"
    )
    parser.add_argument(
        "--led-config",
        type=Path,
        help="LED configuration JSON for runs without an led_config snapshot in session.json",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="worker processes per run (default: %(default)s)",
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate runs that already have ps/"
    )
    args = parser.parse_args(argv)
    if not args.project.is_dir():
        parser.error(f"{args.project} is not a directory")

    status = run_project(
        args.project, args.runs, args.led_config, max(1, args.jobs), args.force
    )

    for name, text in status.items():
        print(f"{name}: {text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
