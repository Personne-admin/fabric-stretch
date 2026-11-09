from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fabsim import texture_tiling as tt
from fabsim.tiling import seam_shift_px


def load_candidates(path: Path) -> list[dict]:
    text = path.read_text().strip()
    records = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines()]
    for i, record in enumerate(records):
        record.setdefault("name", f"candidate_{i:03d}")

    return records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Validate size-search candidates against every state of a package and write previews.",
    )
    ap.add_argument("package", type=Path, help="package directory holding package.json")
    ap.add_argument("orientation", help="warp (ribs cut by the vertical seam) or weft")
    ap.add_argument(
        "candidates",
        type=Path,
        help="JSON Lines from tiling_size_search.py, or a JSON list",
    )
    ap.add_argument("out", type=Path, help="output folder")
    ap.add_argument(
        "--only",
        type=int,
        nargs="+",
        metavar="INDEX",
        help="0-based candidate positions",
    )
    args = ap.parse_args(argv)
    pkg, orientation, out_dir = args.package, args.orientation, args.out
    candidates = load_candidates(args.candidates)
    if args.only is not None:
        candidates = [candidates[i] for i in args.only if 0 <= i < len(candidates)]
    p = json.loads((pkg / "package.json").read_text())
    states = [(b, i, s) for b in ("loading", "release") for i, s in enumerate(p["branches"][b])]
    textures = [np.asarray(Image.open(pkg / s["texture"]).convert("RGB")) for _, _, s in states]
    n_load = len(p["branches"]["loading"])
    picks = {}
    for b, off in (("loading", 0), ("release", n_load)):
        st = p["branches"][b]
        order = sorted(range(len(st)), key=lambda i: st[i]["stretch_ratio"])
        picks.update(
            {
                f"{b} low": off + order[0],
                f"{b} mid": off + order[len(order) // 2],
                f"{b} high": off + order[-1],
            }
        )
    axis = 0 if orientation == "warp" else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for cand in candidates:
        name = cand["name"]
        t0 = time.perf_counter()
        tiles, area, details = tt.find_tile_detailed(
            textures, cand["window"], cand["shift"], cand["interp"], area=cand["area"]
        )
        sec = time.perf_counter() - t0
        h, w = tiles[0].shape[:2]
        shifts = [seam_shift_px(t)[axis] for t in tiles]
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        ref = tiles[picks["loading low"]]
        Image.fromarray(ref).save(d / "tile.png")
        Image.fromarray(np.tile(ref, (3, 3, 1))).save(d / "preview_3x3.png")
        # states sheet: low/mid/high x loading/release, each 3x3 (scaled to a common width)
        cells = []
        for label, idx in picks.items():
            t3 = Image.fromarray(np.tile(tiles[idx], (3, 3, 1)))
            k = 560 / t3.width
            im = t3.resize((560, max(1, int(t3.height * k))))
            ImageDraw.Draw(im).text((4, 3), label, fill=(255, 255, 0))
            cells.append(im)
        ch = max(c.height for c in cells)
        sheet = Image.new("RGB", (560 * 2, ch * 3), "black")
        for k, c in enumerate(cells):
            sheet.paste(c, ((k // 3) * 560, (k % 3) * ch))
        sheet.save(d / "states_3x3.png")
        rec = dict(
            cand,
            tile=[w, h],
            area_out=list(area),
            seconds=round(sec, 1),
            all_states_rel_shift_median=float(np.nanmedian(shifts)),
            all_states_rel_shift_max=float(np.nanmax(shifts)),
            frac_states_shift_gt4=float(np.mean(np.array(shifts) > 4)),
        )
        (d / "metrics.json").write_text(json.dumps(rec, indent=2))
        report.append(rec)
        print(
            name,
            [w, h],
            f"{sec:.0f}s",
            f"median/max shift over {len(states)} states",
            rec["all_states_rel_shift_median"],
            rec["all_states_rel_shift_max"],
            flush=True,
        )
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
