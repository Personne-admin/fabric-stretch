#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

SESSION_FILES = ("session.json", "frames.json", "force.json")


def load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def role_images(run_dir: str, role: str) -> list[str]:
    frames = load_json(os.path.join(run_dir, "frames.json")).get("frames", [])
    key = f"image_{role}"
    names: list[str] = []
    for entry in frames:
        name = entry.get(key) if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            continue

        for rel in (os.path.join("images", name), name):
            if os.path.isfile(os.path.join(run_dir, rel)):
                names.append(rel)
                break

    return names


def cmd_inputs(args: argparse.Namespace) -> int:
    images = role_images(args.run, args.role)
    if not images:
        print(f"error: no '{args.role}' images found for {args.run}", file=sys.stderr)

        return 1

    files = [f for f in SESSION_FILES if os.path.isfile(os.path.join(args.run, f))]
    with open(args.out, "w") as fh:
        fh.write("\n".join(files + images) + "\n")

    print(len(images))

    return 0


def validate_tracks(
    run_dir: str,
    tracks_path: str,
    role: str,
    expect_points: int | None,
    limit: float | None,
) -> dict[str, Any]:
    problems: list[str] = []
    result: dict[str, Any] = {}
    try:
        doc = load_json(tracks_path)
    except (OSError, ValueError) as exc:
        return {**result, "ok": False, "problems": [f"cannot read tracks.json: {exc}"]}

    if doc.get("format") != "fabsim-tracks":
        problems.append(f"format is {doc.get('format')!r}, expected 'fabsim-tracks'")

    frames = doc.get("frames") or []
    images = role_images(run_dir, role)
    result["frames"] = len(frames)
    result["images"] = len(images)
    if len(frames) != len(images):
        problems.append(
            f"{len(frames)} frames in tracks.json but {len(images)} '{role}' images"
        )

    ids = [[p.get("id") for p in f.get("points", [])] for f in frames]
    counts = sorted({len(i) for i in ids})
    result["points"] = counts[0] if len(counts) == 1 else counts
    if not counts or counts == [0]:
        problems.append("no points")
    elif len(counts) > 1:
        problems.append(f"point count differs between frames: {counts}")
    elif any(i != ids[0] for i in ids):
        problems.append("point ids differ between frames")

    if expect_points is not None and counts and counts != [expect_points]:
        problems.append(f"expected {expect_points} points per frame, found {counts}")

    bad = 0
    for f in frames:
        for p in f.get("points", []):
            x, y, c = p.get("x"), p.get("y"), p.get("confidence")
            ok = all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, c)
            )
            if not ok or not 0.0 <= c <= 1.0:
                bad += 1

    if bad:
        problems.append(
            f"{bad} point(s) with non-finite coordinates or confidence outside [0, 1]"
        )

    prov = doc.get("provenance") or {}
    errors = prov.get("reference_errors_px") or {}
    if not errors:
        problems.append("provenance has no reference_errors_px")
    else:
        worst = max(errors.values())
        result["max_reference_error_px"] = round(float(worst), 6)
        if limit is None:
            limit = float(prov.get("reference_error_limit_px", 2.0))

        result["reference_error_limit_px"] = limit
        if not worst < limit:
            problems.append(f"reference error {worst:.4f} px >= limit {limit} px")

    for key in ("adapter_version", "adapter_git_commit", "checkpoint_sha256"):
        if prov.get(key):
            result[key] = prov[key]

    if isinstance(prov.get("runtime"), dict):
        result["gpu"] = prov["runtime"].get("gpu")

    return {**result, "ok": not problems, "problems": problems}


def cmd_validate(args: argparse.Namespace) -> int:
    res = validate_tracks(
        args.run, args.tracks, args.role, args.expect_points, args.limit
    )

    print(json.dumps(res, indent=2))

    return 0 if res["ok"] else 1


def cmd_summary(args: argparse.Namespace) -> int:
    runs = []
    with open(args.entries) as fh:
        for line in fh:
            run, status, detail, seconds, vpath = line.rstrip("\n").split("\t")
            entry: dict[str, Any] = {
                "run": run,
                "status": status,
                "seconds": int(seconds),
            }
            if detail:
                entry["detail"] = detail

            if vpath and os.path.isfile(vpath) and os.path.getsize(vpath) > 0:
                entry["validation"] = load_json(vpath)

            runs.append(entry)

    meta = dict(item.split("=", 1) for item in args.meta)
    counts = {
        s: sum(r["status"] == s for r in runs) for s in ("ok", "failed", "skipped")
    }

    doc = {"run_id": args.run_id, **meta, "counts": counts, "runs": runs}
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect measurement runs and validate tracker outputs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inputs", help="write the list of files to upload for one role")
    p.add_argument("--run", required=True)
    p.add_argument("--role", required=True)
    p.add_argument(
        "--out", required=True, help="file receiving one relative path per line"
    )
    p.set_defaults(func=cmd_inputs)

    p = sub.add_parser("validate", help="check tracks.json against its run")
    p.add_argument("--run", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--role", default="diffuse")
    p.add_argument("--expect-points", type=int)
    p.add_argument(
        "--limit", type=float, help="reference error limit in px (default: the file's)"
    )
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("summary", help="write the dataset summary JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--entries", required=True, help="TSV: run, status, detail, seconds, validation"
    )
    p.add_argument("--meta", nargs="*", default=[], help="extra KEY=VALUE pairs")
    p.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
