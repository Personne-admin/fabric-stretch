from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import dataset


def validate_run(run: Path) -> tuple[list[str], list[str]]:  # (errors, warnings)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        frames = dataset.load_frames(run)
        forces = dataset.load_force(run)
    except (OSError, ValueError, KeyError) as exc:
        return [f"unreadable frames.json/force.json: {exc}"], warnings

    if not frames:
        errors.append("frames.json lists no frames")
    missing_images = missing_force = without_ps = 0

    for frame in frames:
        if not (frame.get("image_diffuse") and dataset.image_path(run, frame["image_diffuse"]).is_file()):
            missing_images += 1
        if not dataset.has_ps(frame):
            without_ps += 1
        elif not all(dataset.image_path(run, frame[key]).is_file() for key in dataset.PS_KEYS):
            missing_images += 1

        if dataset.force_for_frame(frame, forces) is None:
            missing_force += 1

    if missing_images:
        errors.append(f"{missing_images} frame(s) with missing images")

    if missing_force:
        errors.append(f"{missing_force} frame(s) without a valid force reading")

    if without_ps:
        warnings.append(f"{without_ps} frame(s) without photometric-stereo images")

    if not (run / "tracks.json").is_file():
        warnings.append("no tracks.json (not tracked yet)")
    else:
        try:
            json.loads((run / "tracks.json").read_text())
        except ValueError as exc:
            errors.append(f"tracks.json is not valid JSON: {exc}")

    if not (run / "session.json").is_file():
        warnings.append("no session.json")

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fabsim validate", description="Validate measurement runs in a project."
    )
    parser.add_argument(
        "project",
        type=Path,
        help="project folder containing <orientation>_<branch> runs",
    )

    args = parser.parse_args(argv)
    runs = dataset.discover_runs(args.project)
    for name, reason in dataset.ignored_run_folders(args.project):
        print(f"{name}: warn\n  warning: ignored by the pipeline ({reason})")

    if not runs:
        print(
            f"no <orientation>_<loading|release> runs with frames.json in {args.project}",
            file=sys.stderr,
        )

        return 1

    failed = False
    for run in runs:
        errors, warnings = validate_run(run.path)
        failed |= bool(errors)
        status = "ERROR" if errors else ("warn" if warnings else "ok")
        print(f"{run.name}: {status}")
        for text in errors:
            print(f"  error: {text}")
        for text in warnings:
            print(f"  warning: {text}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
