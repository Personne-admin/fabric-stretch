from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import texture_tiling as tt

ALGORITHM = "findTile (SightTex 2.3, 2022) Python port: fabsim.texture_tiling"


@dataclass
class TilingConfig:
    window_size: tuple = (100, 200)  # MATLAB [vertical, horizontal]
    shift: object = 20  # scalar or [left right top down]
    interp_dist: object = 2  # scalar or [vertical, horizontal]
    guidance: object = None  # None = all states, else indices into the state list
    area: object = None  # MATLAB 'area' input (0-based [top left h w], see findTile); None = automatic
    joint_search: bool = False  # score bottom and right edges together (rib-phase aware seam)


def tile_states(textures, cfg: TilingConfig):
    t0 = time.perf_counter()
    guidance = None
    if cfg.guidance:
        bad = [i for i in cfg.guidance if not 0 <= i < len(textures)]
        if bad:
            raise ValueError(
                f"guidance state index {bad[0]} is out of range (the orientation has {len(textures)} states)"
            )
        guidance = [textures[i] for i in cfg.guidance]

    tiles, area, details = tt.find_tile_detailed(
        textures,
        list(cfg.window_size),
        cfg.shift,
        cfg.interp_dist,
        area=cfg.area,
        guidance_images=guidance,
        joint_search=cfg.joint_search,
    )

    elapsed = time.perf_counter() - t0
    H, W = textures[0].shape[:2]
    th, tw = tiles[0].shape[:2]
    top, left = area[0], area[1]
    meta = {
        "enabled": True,
        "algorithm": ALGORITHM,
        "matlab_signature": "findTile(images, window_size, shift, interp_dist, area, guidance_images)",
        "window_size": list(cfg.window_size),
        "shift": cfg.shift,
        "interp_dist": cfg.interp_dist,
        "guidance_images_used": bool(cfg.guidance),
        "guidance_state_indices": list(cfg.guidance) if cfg.guidance else None,
        "area_input_0based": list(cfg.area) if cfg.area else None,
        "states_in_joint_search": len(textures),
        "source_atlas_size_px": [W, H],
        "tile_size_px": [tw, th],
        # MATLAB output area, 1-based [top, left, height, width] (tile is 1 smaller)
        "area_matlab_1based": [int(v) for v in area],
        # Tile pixel (0,0) is atlas pixel (origin_x, origin_y), 0-based.
        "origin_px": [left - 1, top - 1],
        # Texture lookup: tile_px = (atlas_px - origin_px) mod period_px.
        "period_px": [tw, th],
        "origin_uv": [(left - 1) / W, (top - 1) / H],
        "period_uv": [tw / W, th / H],
        "search_seconds": round(elapsed, 2),
        "joint_search": cfg.joint_search,
    }

    return tiles, meta, details


@dataclass(frozen=True)
class Seams:
    area: tuple[int, int, int, int]
    window: tuple[int, int]
    interp: tuple[int, int]
    horizontal: np.ndarray
    vertical: np.ndarray
    atlas_size: tuple[int, int]

    @classmethod
    def from_details(cls, details: dict, atlas_size: tuple[int, int]) -> Seams:
        p = details["params"]
        PH, PV = details["paths"]

        return cls(
            tuple(int(v) for v in details["area"]),
            (p["wh"], p["wv"]),
            (p["idh"], p["idv"]),
            np.asarray(PH, dtype=np.int32),
            np.asarray(PV, dtype=np.int32),
            (int(atlas_size[0]), int(atlas_size[1])),
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            area=np.array(self.area),
            window=np.array(self.window),
            interp=np.array(self.interp),
            horizontal=self.horizontal,
            vertical=self.vertical,
            atlas_size=np.array(self.atlas_size),
        )

    @classmethod
    def load(cls, path: Path) -> Seams:
        with np.load(path) as z:
            return cls(
                tuple(int(v) for v in z["area"]),
                tuple(int(v) for v in z["window"]),
                tuple(int(v) for v in z["interp"]),
                z["horizontal"],
                z["vertical"],
                tuple(int(v) for v in z["atlas_size"]),
            )


def apply_tiling(layer: np.ndarray, seams: Seams, normals: bool = False) -> np.ndarray:
    layer = np.asarray(layer)
    if (layer.shape[1], layer.shape[0]) != seams.atlas_size:
        raise ValueError(
            f"layer size {layer.shape[1]}x{layer.shape[0]} differs from the atlas the seams were "
            f"found on ({seams.atlas_size[0]}x{seams.atlas_size[1]})"
        )

    planes = tt.normalise(layer)
    planes = planes[:, :, None] if planes.ndim == 2 else planes
    y1, y2, x1, x2 = seams.area
    tile = tt.cut_tile(
        planes,
        (seams.horizontal, seams.vertical),
        seams.window[0],
        seams.window[1],
        seams.interp[0],
        seams.interp[1],
        y1,
        y2,
        x1,
        x2,
    )

    if normals:
        n = tile * 2.0 - 1.0
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        tile = n * 0.5 + 0.5

    tile = tt.restore_dtype(tile, layer.dtype)

    return tile[:, :, 0] if layer.ndim == 2 else tile


def quality_checks(tiles, textures):
    shapes = {t.shape for t in tiles}
    if len(shapes) != 1:
        raise ValueError(f"tile shapes differ across states: {shapes}")

    arr = np.asarray(tiles)
    if arr.dtype != np.uint8:
        raise ValueError("tile dtype changed")

    wrap_v, wrap_h, inner_v, inner_h = [], [], [], []
    for t in tiles:
        f = t.astype(np.float32)
        wrap_v.append(np.abs(f[0] - f[-1]).mean())
        wrap_h.append(np.abs(f[:, 0] - f[:, -1]).mean())
        inner_v.append(np.abs(np.diff(f, axis=0)).mean())
        inner_h.append(np.abs(np.diff(f, axis=1)).mean())

    return {
        "tile_shape": list(shapes.pop()),
        "states": len(tiles),
        "wrap_jump_v": float(np.mean(wrap_v)),
        "wrap_jump_h": float(np.mean(wrap_h)),
        "neighbour_diff_v": float(np.mean(inner_v)),
        "neighbour_diff_h": float(np.mean(inner_h)),
    }


def _rgb(a):
    return Image.fromarray(a)


def _repeat(tile, n):
    return np.tile(tile, (n, n, 1))


def write_diagnostics(out_dir, orientation, textures, tiles, meta, details, picks):
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = textures[0].shape[:2]
    area = meta["area_matlab_1based"]
    top, left = area[0], area[1]
    th, tw = tiles[0].shape[:2]
    ref = textures[0]

    # 1) selected area + 2) seams over the source atlas
    ov = _rgb(ref.copy())
    d = ImageDraw.Draw(ov)
    d.rectangle([left - 1, top - 1, left - 1 + tw, top - 1 + th], outline=(255, 255, 0), width=2)
    PH, PV = details["paths"]
    oy, ox = (
        details["crop_origin"][0] - 1,
        details["crop_origin"][1] - 1,
    )  # MATLAB oy, ox
    h0 = th
    w0 = tw  # y2-y1, x2-x1
    for dy, dx, pts, col in (
        (0, 0, PH, (0, 128, 255)),
        (h0, 0, PH, (0, 128, 255)),
        (0, 0, PV, (0, 255, 0)),
        (0, w0, PV, (0, 255, 0)),
    ):
        for r, c in pts:
            x, y = ox + c + dx - 1, oy + r + dy - 1
            if 0 <= x < W and 0 <= y < H:
                ov.putpixel((int(x), int(y)), col)

    ov.save(out_dir / f"{orientation}_area_and_seams.png")

    # 3) tile, 4) 2x2 and 3x3 previews of the reference state
    _rgb(tiles[0]).save(out_dir / f"{orientation}_tile_ref.png")
    _rgb(_repeat(tiles[0], 2)).save(out_dir / f"{orientation}_preview_2x2.png")
    _rgb(_repeat(tiles[0], 3)).save(out_dir / f"{orientation}_preview_3x3.png")

    # 5) low/mid/high loading and release tiled states (2x2 each) in one sheet
    scale = 0.5
    cells = []
    for name, i in picks.items():
        r = _repeat(tiles[i], 2)
        im = _rgb(r).resize((max(1, int(r.shape[1] * scale)), max(1, int(r.shape[0] * scale))))
        ImageDraw.Draw(im).text((6, 6), f"{name} (state {i})", fill=(255, 255, 0))
        cells.append(im)

    cw = max(c.width for c in cells)
    chh = max(c.height for c in cells)
    sheet = Image.new("RGB", (cw * 2, chh * ((len(cells) + 1) // 2)), "black")
    for k, c in enumerate(cells):
        sheet.paste(c, ((k % 2) * cw, (k // 2) * chh))

    sheet.save(out_dir / f"{orientation}_states_2x2.png")


def _hp(v):
    k = max(5, (len(v) // 6) | 1)
    pad = np.concatenate([v[-k:], v, v[:k]])
    sm = np.convolve(pad, np.ones(k) / k, "same")[k:-k]

    return v - sm


def _best_shift(a, b, max_shift):
    a, b = _hp(a), _hp(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    shifts = range(-max_shift, max_shift + 1)

    return max(shifts, key=lambda d: float(np.dot(a, np.roll(b, -d))) / norm)


def seam_shift_px(tile, band=40):
    g = tile.astype(np.float32).mean(axis=2)
    out = []
    for gg in (g, g.T.copy()):
        h, w = gg.shape
        b = min(band, w // 4)
        if b < 6:
            out.append(float("nan"))
            continue

        idx = np.arange(-b, b) % w
        out.append(
            abs(
                _best_shift(
                    gg[:, idx[:b]].mean(axis=1),
                    gg[:, idx[b:]].mean(axis=1),
                    min(28, h // 3),
                )
            )
        )

    return tuple(out)
