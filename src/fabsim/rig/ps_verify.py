from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .. import photometric
from .ps_process import PsOptions, RunData, preprocess_gray, solve_gray

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

NRMSE_PASS = 0.05
NRMSE_WARN = 0.15


class Report:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.lines: list[str] = []
        self.verdicts: list[tuple[str, str]] = []

    def say(self, text: str = ""):
        if not self.quiet:
            print(text)
        self.lines.append(text)

    def verdict(self, status: str, check: str, detail: str):
        self.verdicts.append((status, check))
        self.say(f"[{status}] {check}: {detail}")

    def save(self, path: Path):
        path.write_text("\n".join(self.lines) + "\n")


def ascii_hist(
    rep: Report,
    values: np.ndarray,
    lo: float,
    hi: float,
    bins: int = 10,
    width: int = 40,
    label: str = "",
):
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    total = max(counts.sum(), 1)
    if label:
        rep.say(f"    {label}:")
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:], strict=True):
        bar = "#" * int(round(width * c / max(counts.max(), 1)))
        rep.say(f"    {e0:+.2f}..{e1:+.2f} |{bar:<{width}}| {100.0 * c / total:5.1f}%")


class Solved:
    def __init__(self, run: RunData, idx: int, options: PsOptions, gains=None):
        self.run, self.idx, self.options = run, idx, options
        self.ps_gray, self.albedo, self.normals, _support, self.gains, self.images = solve_gray(
            run, idx, options, gains
        )
        self.mask = valid_mask(self.albedo)
        self.cfg = run.light
        self.frame = run.frames[idx]

    def rendered(self) -> np.ndarray:
        return photometric.render(self.albedo, self.normals, self.cfg, self.gains)

    def diffuse_gray(self) -> np.ndarray | None:
        image = self.run.calibrated(self.run.read_bgr(self.idx, "image_diffuse"), "diffuse")

        return None if image is None else preprocess_gray(image, self.options)


def valid_mask(albedo_raw: np.ndarray) -> np.ndarray:
    med = float(np.median(albedo_raw))

    return albedo_raw > 0.05 * max(med, 1e-3)


def check_lights(rep: Report, cfg: photometric.LightConfig):
    L = photometric.light_matrix(cfg)
    elev, azs = cfg.elevation_deg, list(cfg.azimuths_deg)
    rep.say("-- CHECK 1: light calibration plausibility")
    rep.say(f"  source: {cfg.source}" + (f", gains {np.round(cfg.gains, 3).tolist()}" if cfg.gains else ""))
    rep.say(f"  elevation: {elev:.2f} deg, azimuths: " + ", ".join(f"{a:.2f}" for a in azs) + " deg")
    rep.say("  L =")
    for row in L:
        rep.say("    [{:+.4f} {:+.4f} {:+.4f}]".format(*row))

    cond = float(np.linalg.cond(L))
    status = PASS if cond < 5.0 else (WARN if cond < 15.0 else FAIL)
    rep.verdict(
        status,
        "light matrix conditioning",
        f"cond(L) = {cond:.3f} (pass < 5, fail >= 15)",
    )

    srt = sorted(a % 360.0 for a in azs)
    gaps = [srt[(i + 1) % len(srt)] - srt[i] + (360.0 if i == len(srt) - 1 else 0.0) for i in range(len(srt))]
    ideal = 360.0 / len(azs)
    max_dev = max(abs(g - ideal) for g in gaps)
    status = PASS if max_dev <= 25.0 else (WARN if max_dev <= 45.0 else FAIL)
    rep.verdict(
        status,
        "azimuth spread",
        f"gaps = {', '.join(f'{g:.1f}' for g in gaps)} deg (ideal {ideal:.0f}, max deviation {max_dev:.1f})",
    )

    if not 20.0 <= elev <= 70.0:
        rep.verdict(WARN, "elevation", f"{elev:.1f} deg is outside the comfortable 20..70 range")


def check_normals(rep: Report, normals: np.ndarray, mask: np.ndarray, out_dir: Path, tag: str):
    rep.say(f"-- CHECK 2: normal map sanity ({tag})")
    n_valid = int(mask.sum())
    if n_valid == 0:
        rep.verdict(FAIL, "normals", "mask is empty - nothing to check")

        return

    norms = np.linalg.norm(normals, axis=2)
    dev = np.abs(norms[mask] - 1.0)
    status = PASS if dev.mean() < 0.01 and dev.max() < 0.1 else (WARN if dev.mean() < 0.05 else FAIL)
    rep.verdict(
        status,
        "unit length",
        f"mean |n|-1 deviation {dev.mean():.5f}, max {dev.max():.5f} over {n_valid} valid px",
    )

    nz = normals[..., 2][mask]
    neg_frac = float((nz < 0).mean())
    status = PASS if neg_frac < 0.01 else (WARN if neg_frac < 0.05 else FAIL)
    rep.verdict(
        status,
        "hemisphere",
        f"{100 * neg_frac:.2f}% of valid pixels have negative z (camera-facing should be ~0%)",
    )

    mean_n = normals[mask].mean(axis=0)
    mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-12)
    tilt = float(np.degrees(np.arccos(np.clip(mean_n[2], -1, 1))))
    rep.say(
        f"  mean normal: [{mean_n[0]:+.3f} {mean_n[1]:+.3f} {mean_n[2]:+.3f}]"
        f" (tilt {tilt:.1f} deg from camera axis)"
    )
    if tilt > 20.0:
        rep.verdict(
            WARN,
            "mean tilt",
            f"{tilt:.1f} deg - a flat sample should average near [0,0,1]; "
            "a large tilt suggests an azimuth/elevation calibration bias",
        )
    ascii_hist(rep, nz, -1.0, 1.0, bins=10, label="n_z distribution (valid px)")

    viz = np.clip(normals * 0.5 + 0.5, 0, 1)
    viz = (viz * 255).astype(np.uint8) * mask[..., None].astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{tag}_normals_rgb.png"), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    bins = 128
    hist, _, _ = np.histogram2d(
        normals[..., 1][mask],
        normals[..., 0][mask],
        bins=bins,
        range=[[-1, 1], [-1, 1]],
    )
    hist = np.log1p(hist)
    hist = (255 * hist / max(hist.max(), 1e-9)).astype(np.uint8)
    heat = cv2.applyColorMap(
        cv2.resize(hist, (512, 512), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_INFERNO,
    )
    cv2.circle(heat, (256, 256), 256, (90, 90, 90), 1)
    cv2.putText(heat, "nx ->", (440, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(heat, "ny v", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imwrite(str(out_dir / f"{tag}_normal_direction_hist.png"), heat)
    rep.say(f"  saved: {tag}_normals_rgb.png, {tag}_normal_direction_hist.png")


def check_rerender(rep: Report, solved: Solved, out_dir: Path | None):
    rendered = solved.rendered()
    mask = solved.mask
    results = []
    panels = []
    for i in range(rendered.shape[-1]):
        captured, got = solved.ps_gray[..., i], rendered[..., i]
        diff = got - captured
        rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
        cap_rms = float(np.sqrt(np.mean(captured[mask] ** 2)))
        nrmse = rmse / max(cap_rms, 1e-9)
        corr = float(np.corrcoef(captured[mask].ravel(), got[mask].ravel())[0, 1])
        results.append((rmse, nrmse, corr))
        if out_dir is not None:
            scale = 1.0 / max(float(np.quantile(captured[mask], 0.99)), 1e-6)
            panel = _panel(captured, got, np.abs(diff) * 5.0, scale)
            cv2.putText(
                panel,
                f"light {i + 1}: captured | re-rendered | abs diff x5",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (60, 220, 60),
                2,
            )
            panels.append(panel)
    if out_dir is not None and panels:
        path = out_dir / f"frame{int(solved.frame.get('frame_id', 0)):04d}_rerender_compare.png"
        cv2.imwrite(str(path), np.vstack(panels))
        rep.say(f"  saved: {path.name}")

    return results


def _panel(a: np.ndarray, b: np.ndarray, c: np.ndarray, scale: float) -> np.ndarray:

    def to8(x):
        return (np.clip(x * scale, 0, 1) * 255).astype(np.uint8)

    sep = np.full((a.shape[0], 4), 128, np.uint8)

    return cv2.cvtColor(np.hstack([to8(a), sep, to8(b), sep, to8(c)]), cv2.COLOR_GRAY2BGR)


def diffuse_metrics(solved: Solved, scales=(1, 16)):
    captured = solved.diffuse_gray()
    if captured is None:
        return None
    mask = solved.mask
    pred = solved.rendered().sum(axis=2)
    pm, cm = pred[mask], captured[mask]
    fit_scale = float(np.dot(pm, cm) / max(np.dot(pm, pm), 1e-12))
    per_scale = {}
    for k in scales:
        d = (
            captured
            if k == 1
            else cv2.resize(captured, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
        )
        pr = pred if k == 1 else cv2.resize(pred, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
        s = float((pr * d).sum() / max((pr * pr).sum(), 1e-12))
        nrmse = float(np.sqrt(np.mean((s * pr - d) ** 2)) / max(np.sqrt(np.mean(d**2)), 1e-9))
        corr = float(np.corrcoef(d.ravel(), pr.ravel())[0, 1])
        per_scale[k] = (corr, nrmse)

    return per_scale, captured, pred, fit_scale


def check_diffuse_holdout(rep: Report, solved: Solved, out_dir: Path | None):
    dm = diffuse_metrics(solved, scales=(1, 4, 16))
    if dm is None:
        return None
    per_scale, captured, pred, fit_scale = dm
    corr, nrmse = per_scale[1][0], per_scale[1][1]
    if out_dir is not None:
        scale = 1.0 / max(float(np.quantile(captured[solved.mask], 0.99)), 1e-6)
        panel = _panel(captured, fit_scale * pred, np.abs(fit_scale * pred - captured) * 5.0, scale)
        cv2.putText(
            panel,
            "diffuse (held out): captured | predicted | abs diff x5",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 220, 60),
            2,
        )
        path = out_dir / f"frame{int(solved.frame.get('frame_id', 0)):04d}_diffuse_holdout.png"
        cv2.imwrite(str(path), panel)
        rep.say(f"  saved: {path.name}")
        rep.say("  agreement by scale (downsampled comparison):")
        for k, (c, e) in per_scale.items():
            rep.say(f"    {k:2d}x{k} px blocks: corr {c:.4f}, NRMSE {100 * e:6.2f}%")

    return nrmse, corr


def check_exposure(rep: Report, images: list[np.ndarray]):
    rep.say("-- CHECK 0: capture exposure (raw 8-bit PS images)")
    means = []
    for i, img in enumerate(images):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean, p99 = float(g.mean()), float(np.percentile(g, 99))
        means.append(mean)
        rep.say(f"  light {i + 1}: mean {mean:6.1f}/255, p99 {p99:5.0f}/255, max {int(g.max())}/255")
    lo = min(means)
    status = PASS if lo >= 40.0 else (WARN if lo >= 20.0 else FAIL)
    rep.verdict(
        status,
        "exposure",
        f"darkest light has mean {lo:.1f}/255 (pass >= 40, fail < 20); "
        "underexposed captures make per-pixel PS quantization-noise "
        "dominated regardless of calibration",
    )


def check_albedo(rep: Report, albedo: np.ndarray, mask: np.ndarray, out_dir: Path, tag: str):
    rep.say(f"-- CHECK 4: albedo sanity ({tag})")
    vals = albedo[mask]
    if vals.size == 0:
        rep.verdict(FAIL, "albedo", "mask is empty")

        return
    p1, p50, p99 = np.percentile(vals, [1, 50, 99])
    rep.say(
        f"  range [{albedo.min():.4f}, {albedo.max():.4f}], "
        f"valid-px p1/p50/p99 = {p1:.4f}/{p50:.4f}/{p99:.4f}"
    )
    mask_frac = float(mask.mean())
    rep.say(
        f"  foreground mask covers {100 * mask_frac:.1f}% of the image "
        f"(pipeline rule: albedo > 0.05 * median)"
    )

    zero_frac = float((vals < 1e-4).mean())
    high_frac = float((vals > 0.98 * albedo.max()).mean())
    status = (
        PASS
        if zero_frac < 0.02 and high_frac < 0.02
        else (WARN if zero_frac < 0.10 and high_frac < 0.10 else FAIL)
    )
    rep.verdict(
        status,
        "albedo clipping",
        f"{100 * zero_frac:.2f}% near zero, {100 * high_frac:.2f}% near max within the valid mask",
    )
    if mask_frac < 0.3:
        rep.verdict(
            WARN,
            "mask coverage",
            f"only {100 * mask_frac:.1f}% foreground - check the background/masking behavior",
        )
    ascii_hist(
        rep,
        vals,
        float(vals.min()),
        float(max(vals.max(), 1e-6)),
        bins=10,
        label="albedo distribution (valid px)",
    )
    viz = np.clip(albedo / max(float(np.quantile(vals, 0.99)), 1e-9), 0, 1)
    cv2.imwrite(str(out_dir / f"{tag}_albedo.png"), (viz * 255).astype(np.uint8))
    rep.say(f"  saved: {tag}_albedo.png")


def check_integrability(rep: Report, normals: np.ndarray, mask: np.ndarray, out_dir: Path, tag: str):
    rep.say(f"-- CHECK 5: integrability ({tag})")
    nz = normals[..., 2].copy()
    ok = mask & (nz > 0.05)
    if ok.mean() < 0.05:
        rep.verdict(FAIL, "integrability", "too few pixels with usable n_z")

        return
    nz = np.where(ok, nz, 1.0)
    p = np.where(ok, -normals[..., 0] / nz, 0.0)  # dz/dx
    q = np.where(ok, -normals[..., 1] / nz, 0.0)  # dz/dy

    # Curl of the gradient field should vanish for a real surface.
    curl = np.zeros_like(p)
    curl[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5 - (q[1:-1, 2:] - q[1:-1, :-2]) * 0.5
    curl_rms = float(np.sqrt(np.mean(curl[ok] ** 2)))
    grad_rms = float(np.sqrt(np.mean(p[ok] ** 2 + q[ok] ** 2)))

    # Frankot-Chellappa: project (p, q) onto the nearest integrable field.
    H, W = p.shape
    wx = np.fft.fftfreq(W) * 2.0 * np.pi
    wy = np.fft.fftfreq(H) * 2.0 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    denom = WX**2 + WY**2
    denom[0, 0] = 1.0
    Z = (-1j * WX * np.fft.fft2(p) - 1j * WY * np.fft.fft2(q)) / denom
    Z[0, 0] = 0.0
    z = np.real(np.fft.ifft2(Z))

    zx = np.zeros_like(z)
    zy = np.zeros_like(z)
    zx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) * 0.5
    zy[1:-1, :] = (z[2:, :] - z[:-2, :]) * 0.5
    res = np.sqrt((zx - p) ** 2 + (zy - q) ** 2)
    res_rel = float(np.sqrt(np.mean(res[ok] ** 2)) / max(grad_rms, 1e-9))

    status = PASS if res_rel < 0.3 else (WARN if res_rel < 0.6 else FAIL)
    rep.verdict(
        status,
        "integrability",
        f"gradient residual {100 * res_rel:.1f}% of gradient RMS "
        f"(curl RMS {curl_rms:.4f} vs gradient RMS {grad_rms:.4f})",
    )
    rep.say(
        "  note: fabric weave is genuinely rough; moderate residual is "
        "expected and this check does not gate the overall verdict"
    )

    zm = z - np.median(z[ok])
    span = max(float(np.quantile(np.abs(zm[ok]), 0.99)), 1e-9)
    viz = np.clip((zm / span) * 0.5 + 0.5, 0, 1)
    heat = cv2.applyColorMap((viz * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heat[~mask] = 0
    cv2.imwrite(str(out_dir / f"{tag}_height.png"), heat)
    rep.say(f"  saved: {tag}_height.png")


def discover_run_dirs(paths: list[Path]) -> list[Path]:
    runs = []
    for p in paths:
        p = p.resolve()
        if (p / "frames.json").exists():
            runs.append(p)
            continue
        found = (
            sorted(p.glob("*/frames.json"))
            + sorted(p.glob("*/*/frames.json"))
            + sorted(p.glob("*/*/*/frames.json"))
        )
        runs.extend(fp.parent.resolve() for fp in found)
    seen: set[Path] = set()

    return [r for r in runs if not (r in seen or seen.add(r))]


def _options(args, roi) -> PsOptions:
    return PsOptions(
        gamma=args.gamma,
        ambient_subtract=args.ambient,
        roi=roi,
        estimate_gains=not args.no_gains,
    )


def evaluate_run(run_dir: Path, args, roi) -> dict:
    r: dict = {"dir": run_dir, "error": None, "flags": []}
    try:
        run = RunData(run_dir, args.led_config)
    except (OSError, ValueError) as e:
        r["error"] = f"cannot load run: {e}"

        return r
    ps_keys = run.ps_keys("A")
    if run.mode != "white" or len(ps_keys) != 3:
        r["error"] = f"unsupported (mode={run.mode}, {len(ps_keys)} PS keys)"

        return r
    r["session"] = run.session
    options = _options(args, roi)

    usable = [
        i
        for i, f in enumerate(run.frames)
        if all(f.get(k) and (run_dir / "images" / f[k]).exists() for k in ps_keys)
    ]
    if not usable:
        r["error"] = "no frames with all PS images on disk"

        return r
    n = min(max(1, args.sample), len(usable))
    picks = [usable[i] for i in sorted({int(round(x)) for x in np.linspace(0, len(usable) - 1, n)})]

    means, sats, nrmse1s, corr1s, nrmse16s, negzs = [], [], [], [], [], []
    for idx in picks:
        solved = Solved(run, idx, options)
        grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in solved.images]
        means.append(min(float(g.mean()) for g in grays))
        sats.append(max(float((g >= 250).mean()) for g in grays))
        if solved.mask.sum() == 0:
            continue
        negzs.append(float((solved.normals[..., 2][solved.mask] < 0).mean()))
        dm = diffuse_metrics(solved, scales=(1, 16))
        if dm is not None:
            per_scale = dm[0]
            corr1s.append(per_scale[1][0])
            nrmse1s.append(per_scale[1][1])
            nrmse16s.append(per_scale[16][1])

    if not means:
        r["error"] = "could not evaluate any frame"

        return r
    r["mean8"] = float(np.mean(means))  # darkest light, 8-bit mean
    r["sat"] = float(np.mean(sats))  # worst light, fraction >= 250
    r["negz"] = float(np.mean(negzs)) if negzs else None
    r["nrmse1"] = float(np.mean(nrmse1s)) if nrmse1s else None
    r["corr1"] = float(np.mean(corr1s)) if corr1s else None
    r["nrmse16"] = float(np.mean(nrmse16s)) if nrmse16s else None

    if r["mean8"] < 20.0:
        r["flags"].append("underexp")
    if r["sat"] > 0.01:
        r["flags"].append("clipped")
    if r["nrmse1"] is None:
        r["flags"].append("no-diffuse")

    return r


def run_compare(run_dirs: list[Path], args) -> int:
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    rep = Report()
    rep.say(f"== PS CAPTURE-CONFIG COMPARISON ({len(run_dirs)} runs, {args.sample} sampled frame(s) each)")
    rep.say(
        "  ranking metric: held-out diffuse NRMSE at pixel scale (lower is better); runs flagged "
        "clipped/underexp are not eligible to win"
    )
    rep.say("")

    results = []
    for d in run_dirs:
        print(f"  evaluating {d} ...")
        results.append(evaluate_run(d, args, roi))

    try:
        common = Path(os.path.commonpath([str(r["dir"]) for r in results]))
        if not common.is_dir():
            common = common.parent
    except ValueError:
        common = Path.cwd()

    def name_of(r):
        try:
            return str(r["dir"].relative_to(common)) or r["dir"].name
        except ValueError:
            return str(r["dir"])

    def sess_txt(r):
        s = r.get("session", {})

        return (
            f"led={s.get('led_power', '?')} exp={s.get('cam_exposure_ms', '?')}ms "
            f"settle={s.get('settle_ms', '?')}ms"
        )

    eligible = [r for r in results if r["error"] is None and r["nrmse1"] is not None and not r["flags"]]
    eligible.sort(key=lambda r: r["nrmse1"])

    rep.say(
        f"{'rank':<5}{'run':<36}{'mean8':>6}{'sat%':>7}{'NRMSE%':>8}{'16x%':>7}{'corr':>7}{'negz%':>7}  "
        "status"
    )

    def fmt(v, f):
        return f.format(v) if v is not None else "n/a"

    for i, r in enumerate(eligible + [r for r in results if r not in eligible]):
        if r["error"] is not None:
            rep.say(f"{'-':<5}{name_of(r):<36}  error: {r['error']}")
            continue
        status = ",".join(r["flags"]) if r["flags"] else ("BEST" if eligible and r is eligible[0] else "ok")
        rank = str(i + 1) if r in eligible else "-"
        rep.say(
            f"{rank:<5}{name_of(r):<36}{r['mean8']:>6.1f}{100 * r['sat']:>7.2f}"
            f"{fmt(r['nrmse1'], '{:.1%}').rstrip('%'):>8}{fmt(r['nrmse16'], '{:.1%}').rstrip('%'):>7}"
            f"{fmt(r['corr1'], '{:.3f}'):>7}{fmt(r['negz'], '{:.2%}').rstrip('%'):>7}  {status}"
        )
    rep.say("")

    rep.say("== RECOMMENDATION")
    if not eligible:
        rep.say(
            "  no eligible run: every config is flagged (see status column). Aim for darkest-light mean in "
            "roughly 40..180 of 255 with saturation below 1%, then re-capture."
        )
        outcome = 1
    else:
        best = eligible[0]
        rep.say(f"  best config: {name_of(best)}  ({sess_txt(best)})")
        rep.say(
            f"  held-out diffuse NRMSE {100 * best['nrmse1']:.1f}% at pixel scale "
            f"({100 * best['nrmse16']:.1f}% at 16x16), darkest-light mean {best['mean8']:.0f}/255, "
            f"saturation {100 * best['sat']:.2f}%"
        )
        if best["nrmse1"] < 0.20:
            rep.say(
                "  quality: good - per-pixel normals are consistent enough to build relighting on. "
                "Use this config."
            )
            outcome = 0
        elif best["nrmse1"] < 0.40:
            rep.say(
                "  quality: marginal - coarse-scale geometry is fine but per-pixel normals are noisy. "
                "Usable for low-frequency relighting; push exposure/LED power higher (or average multiple "
                "shots) for per-pixel work."
            )
            outcome = 0
        else:
            rep.say(
                "  quality: insufficient - even the best config is noise-dominated at pixel scale. "
                "Increase exposure/LED power toward darkest-light mean 40+/255, avoid saturation, or "
                "capture more than 8 bits."
            )
            outcome = 1
        rep.say(f"  for details run: fabsim ps-verify {best['dir']}")

    out_path = common / "ps_compare.txt"
    rep.save(out_path)
    print(f"\n  comparison written to {out_path}")

    return outcome


def run_detail(run_dir: Path, args) -> int:
    run_dir = run_dir.resolve()
    out_dir = run_dir / "ps_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    options = _options(args, roi)

    rep = Report()
    rep.say(f"== PS VERIFICATION: {run_dir}")
    try:
        run = RunData(run_dir, args.led_config)
    except (OSError, ValueError) as e:
        sys.exit(f"ERROR: cannot load {run_dir}: {e}")
    frames = run.frames
    ps_keys = run.ps_keys("A")
    rep.say(
        f"  mode: {run.mode} | frames: {len(frames)} | ps keys: "
        + (", ".join(ps_keys) if ps_keys else "(none)")
    )
    if run.mode != "white" or len(ps_keys) != 3:
        sys.exit(
            "ERROR: this verifier handles white-light PS runs with exactly 3 PS images per frame; "
            f"found mode={run.mode}, {len(ps_keys)} keys"
        )
    rep.say(
        f"  settings: gamma={options.gamma}, ambient={options.ambient_subtract}, roi={roi}, "
        f"gain estimation={'on' if options.estimate_gains else 'off'}"
    )
    rep.say("")

    check_lights(rep, run.light)
    rep.say("")

    if not 0 <= args.frame < len(frames):
        sys.exit(f"ERROR: --frame {args.frame} out of range 0..{len(frames) - 1}")
    frame = frames[args.frame]
    fid = int(frame.get("frame_id", args.frame))
    tag = f"frame{fid:04d}"
    rep.say(
        f"== detail frame {args.frame} (frame_id {fid}, "
        f"displacement {frame.get('displacement_mm', 0.0):.1f} mm)"
    )
    try:
        solved = Solved(run, args.frame, options)
    except RuntimeError as e:
        sys.exit(f"ERROR: frame {args.frame}: {e}")
    rep.say(f"  LED gains used: {np.round(solved.gains, 3).tolist()}")

    check_exposure(rep, [run.read_bgr(args.frame, k) for k in ps_keys])
    rep.say("")
    check_normals(rep, solved.normals, solved.mask, out_dir, tag)
    rep.say("")
    rep.say("-- CHECK 3: re-rendering self-consistency (the key check)")
    res = check_rerender(rep, solved, out_dir)
    for i, (rmse, nrmse, corr) in enumerate(res):
        rep.say(f"  light {i + 1}: RMSE {rmse:.5f}, NRMSE {100 * nrmse:.2f}%, corr {corr:.4f}")
    detail_nrmse = float(np.mean([r[1] for r in res]))
    status = PASS if detail_nrmse < NRMSE_PASS else (WARN if detail_nrmse < NRMSE_WARN else FAIL)
    rep.verdict(
        status,
        "re-rendering (detail frame)",
        f"mean NRMSE {100 * detail_nrmse:.2f}% (pass < {100 * NRMSE_PASS:.0f}%, "
        f"fail >= {100 * NRMSE_WARN:.0f}%)",
    )
    rep.say(
        "  CAVEAT: with exactly 3 lights the PS solve is exactly determined, so near-zero error here is "
        "expected by construction; it confirms consistent preprocessing and light bookkeeping, NOT correct "
        "normals. The held-out diffuse check below is the discriminative test."
    )
    rep.say("")
    rep.say("-- CHECK 3b: held-out diffuse prediction (discriminative)")
    diffuse_detail = check_diffuse_holdout(rep, solved, out_dir)
    if diffuse_detail is None:
        rep.say(
            "  no image_diffuse in this run - skipped (per-light check is then consistency-only; "
            "rely on normals/integrability)"
        )
    else:
        d_nrmse, d_corr = diffuse_detail
        status = PASS if d_nrmse < 0.20 else (WARN if d_nrmse < 0.40 else FAIL)
        rep.verdict(
            status,
            "held-out diffuse prediction",
            f"NRMSE {100 * d_nrmse:.2f}% after exposure fit, corr {d_corr:.4f} (pass < 20%, fail >= 40%; "
            "the diffuse capture's light geometry only approximates the summed PS lights, so this is "
            "a looser bound)",
        )
    rep.say("")
    check_albedo(rep, solved.albedo, solved.mask, out_dir, tag)
    rep.say("")
    if args.integrability:
        check_integrability(rep, solved.normals, solved.mask, out_dir, tag)
        rep.say("")

    rep.say(f"== all-frame summary ({len(frames)} frames)")
    rows = []
    for i in range(len(frames)):
        try:
            s = Solved(run, i, options, solved.gains)
        except RuntimeError as e:
            rep.say(f"  frame {i}: {e} - skipped")
            continue
        if s.mask.sum() == 0:
            rep.say(f"  frame {i}: empty mask - skipped")
            continue
        nrmse = float(np.mean([r[1] for r in check_rerender(rep, s, None)]))
        neg = float((s.normals[..., 2][s.mask] < 0).mean())
        dif = check_diffuse_holdout(rep, s, None)
        rows.append((i, nrmse, neg, dif[0] if dif else None))
        dif_txt = f"{100 * dif[0]:6.2f}%" if dif else "   n/a"
        rep.say(
            f"  frame {i:3d}: per-light NRMSE {100 * nrmse:6.2f}% | diffuse NRMSE {dif_txt} | "
            f"neg-z {100 * neg:5.2f}% | albedo p50 {np.median(s.albedo[s.mask]):.4f}"
        )
    rep.say("")

    rep.say("== OVERALL VERDICT")
    n_fail = sum(1 for s, _ in rep.verdicts if s == FAIL)
    n_warn = sum(1 for s, _ in rep.verdicts if s == WARN)
    if rows:
        mean_nrmse = float(np.mean([r[1] for r in rows]))
        rep.say(
            f"  per-light reconstruction: mean NRMSE {100 * mean_nrmse:.2f}% across {len(rows)} frames "
            "(consistency only with 3 lights)"
        )
        dif_vals = [r[3] for r in rows if r[3] is not None]
        key_nrmse = float(np.mean(dif_vals)) if dif_vals else mean_nrmse
        key_name = "held-out diffuse" if dif_vals else "per-light"
        if dif_vals:
            worst = max((r for r in rows if r[3] is not None), key=lambda r: r[3])
            rep.say(
                f"  held-out diffuse: mean NRMSE {100 * key_nrmse:.2f}%, worst frame {worst[0]} at "
                f"{100 * worst[3]:.2f}%"
            )
        good = 0.20 if dif_vals else NRMSE_PASS
        bad = 0.40 if dif_vals else NRMSE_WARN
        if key_nrmse < good and n_fail == 0:
            rep.say(
                f"  VERDICT: PS output looks trustworthy. The {key_name} reconstruction is close and no "
                "fundamental check failed; good enough to build relighting on."
            )
        elif key_nrmse < bad and n_fail == 0:
            rep.say(
                f"  VERDICT: usable but imperfect. The {key_name} reconstruction error is moderate - "
                "relighting will look plausible but expect artifacts where the error concentrates (check "
                "the comparison images). Review the WARN items above."
            )
        else:
            rep.say(
                f"  VERDICT: NOT trustworthy yet. The {key_name} reconstruction error is high or a "
                "fundamental check failed - fix calibration/processing before building relighting."
            )
    else:
        rep.say("  VERDICT: could not evaluate any frame - see errors above.")
    rep.say(f"  checks: {len(rep.verdicts)} total, {n_warn} WARN, {n_fail} FAIL")
    rep.say(f"  diagnostics written to {out_dir}")

    rep.save(out_dir / "summary.txt")

    return 1 if n_fail > 0 else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fabsim ps-verify",
        description="Photometric-stereo verification and capture-config comparison",
        epilog="One run folder: detailed verification with diagnostic images in <run>/ps_verify/. Several "
        "folders (or a parent folder of test captures): compact comparison table ranking the "
        "LED/exposure configs by PS quality, best config recommended.",
    )
    ap.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="run folder(s) with frames.json, or parent folder(s) to scan",
    )
    ap.add_argument(
        "--frame",
        type=int,
        default=0,
        help="frame index to inspect in detail (default 0)",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="force the comparison table even for a single run",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=3,
        help="frames sampled per run in comparison mode (default 3)",
    )
    ap.add_argument(
        "--integrability",
        action="store_true",
        help="also run the optional height-integration check (detail mode)",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="exponent applied to the gray images (default 1: the camera bytes are linear)",
    )
    ap.add_argument(
        "--ambient",
        action="store_true",
        help="subtract the 1st percentile of every image",
    )
    ap.add_argument("--no-gains", action="store_true", help="do not estimate the LED gains")
    ap.add_argument(
        "--roi",
        type=str,
        default=None,
        help="white-calibration ROI as X,Y,W,H (default none)",
    )
    ap.add_argument(
        "--led-config",
        type=Path,
        default=None,
        help="led_config.json used when a run has no led_config snapshot in session.json",
    )

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dirs = discover_run_dirs(args.paths)
    if not run_dirs:
        sys.exit("ERROR: no run folders (containing frames.json) found under the given path(s)")
    if len(run_dirs) == 1 and not args.compare:
        return run_detail(run_dirs[0], args)

    return run_compare(run_dirs, args)


if __name__ == "__main__":
    sys.exit(main())
