from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fabsim import texture_tiling as tt
from fabsim.tiling import seam_shift_px

PICKS = None
ORIENTATION = None


def load_picks(pkg: Path):
    p = json.loads((pkg / "package.json").read_text())
    files = []
    for branch in ("loading", "release"):
        st = p["branches"][branch]
        order = sorted(range(len(st)), key=lambda i: st[i]["stretch_ratio"])
        for i in (order[0], order[len(order) // 2], order[-1]):
            files.append(st[i]["texture"])

    return [np.asarray(Image.open(pkg / f).convert("RGB")) for f in files], files


def init(pkg, orientation):
    global PICKS, ORIENTATION
    PICKS, _ = load_picks(Path(pkg))
    ORIENTATION = orientation


def sample(rng, orientation, H, W):
    area_h = rng.randrange(48, 260 if orientation == "weft" else 200, 2)
    area_w = rng.randrange(60, 520, 2)
    shift = rng.choice([0, 4, 8, 12, 20, 30])
    idist = rng.choice([1, 2, 3])
    wh = rng.randrange(
        8, max(12, min(90, int(area_h * 0.55))), 2
    )  # height of top/bottom windows
    wv = rng.randrange(
        8, max(12, min(120, int(area_w * 0.45))), 2
    )  # width of left/right windows
    wt, wl = wh // 2, wv // 2
    top_lo, left_lo = idist + wt, idist + wl
    top_hi = H - area_h - shift - idist - wh
    left_hi = W - area_w - shift - idist - wv
    if top_hi <= top_lo or left_hi <= left_lo:
        return None

    return dict(
        window=[wv, wh],
        shift=shift,
        interp=idist,
        area=[
            rng.randrange(top_lo, top_hi),
            rng.randrange(left_lo, left_hi),
            area_h,
            area_w,
        ],
    )


def evaluate(params):
    t0 = time.perf_counter()
    try:
        tiles, area, _ = tt.find_tile(
            PICKS,
            params["window"],
            params["shift"],
            params["interp"],
            area=params["area"],
        )
    except (ValueError, RuntimeError):
        return None
    sec = time.perf_counter() - t0
    axis = 0 if ORIENTATION == "warp" else 1  # seam that cuts the ribs
    shifts = [seam_shift_px(t) for t in tiles]
    rel = [s[axis] for s in shifts]
    oth = [s[1 - axis] for s in shifts]
    h, w = tiles[0].shape[:2]

    return dict(
        params,
        tile=[w, h],
        area_out=list(area),
        rel_shift_med=float(np.nanmedian(rel)),
        rel_shift_max=float(np.nanmax(rel)),
        other_shift_med=float(np.nanmedian(oth)),
        seconds=round(sec, 2),
    )


def _work(args):
    return evaluate(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("orientation")
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    picks, _ = load_picks(Path(a.package))
    H, W = picks[0].shape[:2]
    rng = random.Random(a.seed)
    cands = [
        c for c in (sample(rng, a.orientation, H, W) for _ in range(a.n * 3)) if c
    ][: a.n]
    t0 = time.perf_counter()
    done = 0
    with (
        mp.Pool(a.procs, initializer=init, initargs=(a.package, a.orientation)) as pool,
        open(a.out, "w") as f,
    ):
        for r in pool.imap_unordered(_work, cands, chunksize=2):
            done += 1
            if r:
                f.write(json.dumps(r) + "\n")
                f.flush()
    print(f"{done} candidates in {time.perf_counter() - t0:.0f}s")
