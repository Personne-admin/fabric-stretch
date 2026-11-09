from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fabsim import tiling as ts

_METRIC_KEYS = (
    "seam_err0",
    "seam_err_best",
    "interior_err0",
    "ratio0",
    "median_abs_dy",
    "frac_blocks_dy_gt5",
)


def seam_metric(tile, axis="x", block=56, stride=28, maxd=30, strip=3, gap_ref_cols=None):
    g = tile.astype(np.float32).mean(axis=2)
    if axis == "y":
        g = g.T
    H, W = g.shape
    maxd = min(maxd, H // 4)  # small tiles: shrink the search window and the blocks to fit
    block = min(block, H - 2 * maxd - 1)
    if gap_ref_cols is None:
        gap_ref_cols = [int(W * f) for f in (0.2, 0.4, 0.6)]
    gap_ref_cols = [c for c in gap_ref_cols if strip <= c < W - strip]
    if block < 8 or not gap_ref_cols:
        return dict.fromkeys(_METRIC_KEYS, float("nan"))

    def pair(left_cols, right_cols):
        L = g[:, left_cols].mean(axis=1)
        R = g[:, right_cols].mean(axis=1)
        errs0, dys, errsb = [], [], []
        for r in range(maxd, H - block - maxd, stride):
            a = L[r : r + block]
            e = [np.abs(a - R[r + d : r + d + block]).mean() for d in range(-maxd, maxd + 1)]
            errs0.append(e[maxd])
            k = int(np.argmin(e))
            dys.append(k - maxd)
            errsb.append(e[k])

        return np.mean(errs0), np.array(dys), np.mean(errsb)

    s0, dys, sb = pair(slice(W - strip, W), slice(0, strip))
    i0 = float(np.mean([pair(slice(c - strip, c), slice(c, c + strip))[0] for c in gap_ref_cols]))

    return {
        "seam_err0": float(s0),
        "seam_err_best": float(sb),
        "interior_err0": i0,
        "ratio0": float(s0 / i0),
        "median_abs_dy": float(np.median(np.abs(dys))),
        "frac_blocks_dy_gt5": float(np.mean(np.abs(dys) > 5)),
    }


def load_states(pkg):
    p = json.loads((pkg / "package.json").read_text())
    names = [s["texture"] for b in ("loading", "release") for s in p["branches"][b]]

    return p, names, [np.asarray(Image.open(pkg / n).convert("RGB")) for n in names]


def picks_from(p):
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

    return picks


def run(orientation, out_root, name, cfg, textures, picks):
    out = out_root / name
    t0 = time.perf_counter()
    tiles, meta, details = ts.tile_states(textures, cfg)
    elapsed = time.perf_counter() - t0
    ts.write_diagnostics(out, orientation, textures, tiles, meta, details, picks)
    two = np.tile(tiles[0], (2, 2, 1))
    th, tw = tiles[0].shape[:2]
    x0 = max(0, tw - 250)
    Image.fromarray(two[:, x0 : tw + 250]).resize((1000, two.shape[0] * 2), Image.LANCZOS).save(
        out / f"{orientation}_vseam_zoom.png"
    )
    m = [seam_metric(tiles[i]) for i in picks.values()]
    summary = {k: float(np.mean([x[k] for x in m])) for k in m[0]}
    shifts = [ts.seam_shift_px(tiles[i]) for i in picks.values()]
    summary["seam_shift_px"] = {
        "x_seam_per_state": [float(x) for x, _ in shifts],
        "y_seam_per_state": [float(y) for _, y in shifts],
        "x_seam_median": float(np.nanmedian([x for x, _ in shifts])),
        "y_seam_median": float(np.nanmedian([y for _, y in shifts])),
    }
    summary["per_state"] = dict(zip(picks, m, strict=True))
    res = {
        "name": name,
        "tile_size_px": meta["tile_size_px"],
        "area_matlab_1based": meta["area_matlab_1based"],
        "window_size": list(cfg.window_size),
        "shift": cfg.shift,
        "interp_dist": cfg.interp_dist,
        "guidance": cfg.guidance,
        "seconds": round(elapsed, 1),
        "seam": summary,
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2))

    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Measure wrap-seam quality across material-package tiling configurations.",
    )
    ap.add_argument("package")
    ap.add_argument("out_root")
    ap.add_argument("name")
    ap.add_argument("--window-size", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--shift", type=int, nargs="+", default=[20])
    ap.add_argument("--interp-dist", type=int, nargs="+", default=[2])
    ap.add_argument(
        "--area",
        type=int,
        nargs=4,
        default=None,
        help="MATLAB area input, 0-based [top left height width]",
    )
    ap.add_argument(
        "--guidance-picks",
        action="store_true",
        help="guidance = the 6 low/mid/high Loading+Release states",
    )
    a = ap.parse_args()
    pkg = Path(a.package)
    p, names, textures = load_states(pkg)
    picks = picks_from(p)

    def one(v):
        return v[0] if len(v) == 1 else list(v)

    cfg = ts.TilingConfig(
        tuple(a.window_size),
        one(a.shift),
        one(a.interp_dist),
        sorted(picks.values()) if a.guidance_picks else None,
        a.area,
    )
    r = run(p.get("orientation", "warp"), Path(a.out_root), a.name, cfg, textures, picks)
    print(json.dumps({k: r[k] for k in r if k != "seam"} | {"seam_shift_px": r["seam"]["seam_shift_px"]}))
