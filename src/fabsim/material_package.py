from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from . import dataset, ps_frames
from . import photometric as ph

FORMAT = "fabsim-material-rgb"
VERSION = 3
#: Tracker confidence below which a point does not count as tracked.
CONFIDENCE = 0.5

#: Largest tolerated registration error of the Release reference grid (95th percentile, px).
MAX_REFERENCE_ERROR_PX = 2.0
STRETCH_DEFINITION = "median tracked edge ratio along the pull axis"
#: Loading states (at most) whose readings are fitted for the tare, and the stretch up to which they count.
TARE_STATES = 10
TARE_MAX_STRETCH = 1.03
TARE_METHOD = "Release readings are re-referenced to the Loading zero_counts first"

#: A branch's force is flagged unusable below this span / noise ratio or rank correlation with stretch.
FORCE_MIN_SNR = 3.0
FORCE_MIN_SPEARMAN = 0.5
#: Fewest states for the force quality figures.
FORCE_QUALITY_MIN_STATES = 8
ALBEDO_ENCODING = "linear RGB, 16-bit PNG; albedo = v/65535*albedo_scale"


#: Factor on the measured percentile so that the pixels at the percentile itself stay below full scale.
ALBEDO_HEADROOM = 1.001


def albedo_scale_basis(states: int) -> str:
    return (
        "smallest value that at most 0.1 % of the brightest-channel albedo pixels exceed, over the atlas "
        f"of all {states} states, plus 0.1 % headroom"
    )


EXPOSURE_BASIS = (
    "albedo is multiplied by exposure_gain = mean(diffuse luma) / mean(albedo * max(n_z, 0)) measured on "
    "the Loading reference frame, so a relight straight on with intensity 1 and ambient 0 is as bright "
    "as the diffuse photograph"
)
NORMAL_CORRECTION = (
    "per state, a plane fitted to the x and y components of the normal over the whole atlas is removed "
    "and the normals renormalised, normal_tilt_deg of a state is the tilt of its mean normal "
    "before the correction"
)
NORMAL_ENCODING = "16-bit RGB PNG; n = 2*v/65535-1; x=atlas column/right, y=atlas row UP, z=toward camera"
PS_FRAME = "atlas axes = camera image axes: x=atlas column/right, y=atlas row UP (i.e. -row), z=toward camera"
SEAMS_FILE = "tiling_seams.npz"

_PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 6]


@dataclass(frozen=True)
class Tile:
    col: int
    row: int
    cols: int
    rows: int
    bounds: tuple[float, float, float, float]
    size: tuple[int, int]


def grid_frames(track_file: Path):
    data = json.loads(track_file.read_text())
    frames = {}
    for frame in data["frames"]:
        points = {int(p["id"].removeprefix("grid_")): p for p in frame["points"]}
        n = len(points)
        if set(points) != set(range(n)):
            raise ValueError("tracker IDs must be contiguous grid_0..grid_N")

        frames[int(frame["frame_id"])] = points

    ref_id = int(data.get("provenance", {}).get("reference_frame", 0))
    ref = frames[ref_id]
    first_y = ref[0]["y"]
    cols = sum(abs(ref[i]["y"] - first_y) < 1e-4 for i in range(len(ref)))
    if cols < 2 or len(ref) % cols or len(ref) // cols < 2:
        raise ValueError("tracker topology is not a rectangular grid")

    return frames, ref_id, cols, len(ref) // cols


def _xy(points) -> np.ndarray:
    return np.array([[points[i]["x"], points[i]["y"]] for i in range(len(points))], dtype=np.float32)


def largest_rectangle(mask: np.ndarray):
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, 0, 0, 0, 0)

    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)

        stack = []
        for col in range(mask.shape[1] + 1):
            height = int(heights[col]) if col < mask.shape[1] else 0
            start = col
            while stack and stack[-1][1] > height:
                left, h = stack.pop()
                area = h * (col - left)
                if area > best[0]:
                    best = (area, left, row - h + 1, col - left, h)

                start = left

            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))

    if not best[0]:
        raise ValueError("no common valid tracker cell")

    return best[1:]


def register_references(loading_image: np.ndarray, release_image: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.005)
    a, da = sift.detectAndCompute(cv2.cvtColor(loading_image, cv2.COLOR_RGB2GRAY), None)
    b, db = sift.detectAndCompute(cv2.cvtColor(release_image, cv2.COLOR_RGB2GRAY), None)
    if da is None or db is None:
        raise ValueError("reference images have no matchable material features")

    matches = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
    if len(matches) < 30:
        raise ValueError("reference images do not establish a shared physical patch")

    source = np.float32([a[m.queryIdx].pt for m in matches])
    target = np.float32([b[m.trainIdx].pt for m in matches])
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or mask is None:
        raise ValueError("reference image registration failed")

    inliers = source[mask.ravel() != 0]
    area = cv2.contourArea(cv2.convexHull(inliers)) / (loading_image.shape[0] * loading_image.shape[1])

    if len(inliers) < 30 or area < 0.25:
        raise ValueError("reference matches do not cover enough of the physical patch")

    return homography


def grid_coordinates(points, cols, rows, xy):
    shape = xy.shape[:-1]
    flat = xy.reshape(-1, 2).astype(np.float32)
    corners = np.float32(
        [
            [points[0]["x"], points[0]["y"]],
            [points[cols - 1]["x"], points[cols - 1]["y"]],
            [
                points[(rows - 1) * cols + cols - 1]["x"],
                points[(rows - 1) * cols + cols - 1]["y"],
            ],
            [points[(rows - 1) * cols]["x"], points[(rows - 1) * cols]["y"]],
        ]
    )

    target = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])
    guess = cv2.perspectiveTransform(flat[None], cv2.getPerspectiveTransform(corners, target))[0]

    gx, gy = guess[:, 0], guess[:, 1]
    grid = _xy(points)

    for _ in range(5):
        cx = np.clip(np.floor(gx).astype(int), 0, cols - 2)
        cy = np.clip(np.floor(gy).astype(int), 0, rows - 2)
        idx = cy * cols + cx
        u, v = gx - cx, gy - cy
        p00, p10, p01, p11 = (
            grid[idx],
            grid[idx + 1],
            grid[idx + cols],
            grid[idx + cols + 1],
        )
        predicted = (
            ((1 - u) * (1 - v))[:, None] * p00
            + (u * (1 - v))[:, None] * p10
            + ((1 - u) * v)[:, None] * p01
            + (u * v)[:, None] * p11
        )
        du = (1 - v)[:, None] * (p10 - p00) + v[:, None] * (p11 - p01)
        dv = (1 - u)[:, None] * (p01 - p00) + u[:, None] * (p11 - p10)
        error = flat - predicted
        det = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
        safe = np.where(np.abs(det) > 1e-6, det, 1.0)
        gx += np.clip((error[:, 0] * dv[:, 1] - error[:, 1] * dv[:, 0]) / safe, -1.0, 1.0)
        gy += np.clip((du[:, 0] * error[:, 1] - du[:, 1] * error[:, 0]) / safe, -1.0, 1.0)

    return np.stack((gx, gy), axis=-1).reshape(*shape, 2)


def sample_grid(points, cols, rows, grid_xy):
    gx, gy = grid_xy[..., 0], grid_xy[..., 1]
    cx = np.clip(np.floor(gx).astype(int), 0, cols - 2)
    cy = np.clip(np.floor(gy).astype(int), 0, rows - 2)
    u, v = gx - cx, gy - cy
    idx = cy * cols + cx
    xy = _xy(points)

    return (
        ((1 - u) * (1 - v))[..., None] * xy[idx]
        + (u * (1 - v))[..., None] * xy[idx + 1]
        + ((1 - u) * v)[..., None] * xy[idx + cols]
        + (u * v)[..., None] * xy[idx + cols + 1]
    )


def _checked(mapping: np.ndarray, shape, what: str) -> np.ndarray:
    if (
        mapping[..., 0].min() < 0
        or mapping[..., 0].max() > shape[1] - 1
        or mapping[..., 1].min() < 0
        or mapping[..., 1].max() > shape[0] - 1
    ):
        raise ValueError(f"{what} mapping leaves image support")

    return mapping.astype(np.float32)


def loading_map(points, cols: int, tile: Tile, shape) -> np.ndarray:
    w, h = tile.size
    xx = np.linspace(0, tile.cols, w, dtype=np.float32)
    yy = np.linspace(0, tile.rows, h, dtype=np.float32)
    ci = np.minimum(xx.astype(int), tile.cols - 1)
    ri = np.minimum(yy.astype(int), tile.rows - 1)
    u = (xx - ci)[None, :]
    v = (yy - ri)[:, None]
    ids = (ri[:, None] + tile.row) * cols + (ci[None, :] + tile.col)
    xy = _xy(points)
    p00, p10, p01, p11 = xy[ids], xy[ids + 1], xy[ids + cols], xy[ids + cols + 1]
    mapping = (
        ((1 - u) * (1 - v))[..., None] * p00
        + (u * (1 - v))[..., None] * p10
        + ((1 - u) * v)[..., None] * p01
        + (u * v)[..., None] * p11
    )

    return _checked(mapping, shape, "tile")


def release_map(points, cols: int, rows: int, release_grid: np.ndarray, shape) -> np.ndarray:
    return _checked(sample_grid(points, cols, rows, release_grid), shape, "shared atlas")


def warp(image: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image,
        mapping[..., 0],
        mapping[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def remap_grid(image: np.ndarray, points, cols: int, tile: Tile):
    return warp(image, loading_map(points, cols, tile, image.shape))


def remap_shared(image, points, cols, rows, release_grid):
    return warp(image, release_map(points, cols, rows, release_grid, image.shape))


def stretch_ratio(points, reference, cols: int, tile: Tile, axis: str):
    ratios = []
    for row in range(tile.row, tile.row + tile.rows + 1):
        for col in range(tile.col, tile.col + tile.cols + 1):
            if axis == "horizontal" and col < tile.col + tile.cols:
                i, j = row * cols + col, row * cols + col + 1
            elif axis == "vertical" and row < tile.row + tile.rows:
                i, j = row * cols + col, (row + 1) * cols + col
            else:
                continue

            a = np.array([points[j]["x"] - points[i]["x"], points[j]["y"] - points[i]["y"]])

            b = np.array(
                [
                    reference[j]["x"] - reference[i]["x"],
                    reference[j]["y"] - reference[i]["y"],
                ]
            )
            if np.linalg.norm(b) > 1e-6:
                ratios.append(np.linalg.norm(a) / np.linalg.norm(b))

    return float(np.median(ratios))


def _axis_stretch(current: np.ndarray, baseline: np.ndarray, axis: str) -> float:
    along = 1 if axis == "horizontal" else 0
    lengths = np.linalg.norm(np.diff(baseline, axis=along), axis=-1)
    now = np.linalg.norm(np.diff(current, axis=along), axis=-1)

    return float(np.median(now[lengths > 1e-6] / lengths[lengths > 1e-6]))


@dataclass
class RunData:
    path: Path
    branch: str
    tracks: dict[int, dict]
    cols: int
    rows: int
    frames: list[dict]  # tracked frames with a diffuse image and a valid force reading, chronological
    forces: dict[int, dict]
    reference_id: int
    reference_image: np.ndarray
    clamp_width_m: float | None
    skipped: list[str] = field(default_factory=list)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read image {path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_run(run: dataset.Run) -> RunData:
    path = run.path
    tracks, tracker_ref, cols, rows = grid_frames(path / "tracks.json")
    frames_doc = json.loads((path / "frames.json").read_text())
    forces = dataset.load_force(path)
    usable, untracked, no_force = [], 0, []

    for position, frame in enumerate(frames_doc["frames"]):
        fid = dataset.frame_id(frame, position)
        image = frame.get("image_diffuse")
        if fid not in tracks or not image or not dataset.image_path(path, image).is_file():
            untracked += 1
        elif dataset.force_for_frame(frame, forces) is None:
            no_force.append(f"{run.branch} frame {fid}: no valid force reading")
        else:
            usable.append({**frame, "frame_id": fid})

    skipped = list(no_force)
    if untracked:
        skipped.append(f"{run.branch}: {untracked} frame(s) without tracking or diffuse image")

    if not usable:
        raise ValueError(f"{run.name}: no tracked diffuse frames with a valid force reading")

    if run.branch == "loading":
        reference = next((f for f in usable if f["frame_id"] == tracker_ref), None)
        if reference is None:
            raise ValueError(
                f"{run.name}: tracker reference frame {tracker_ref} has no image or force reading"
            )
    else:
        reference = usable[-1]  # the end of the release is the relaxed state

    session = dataset.load_session(path)
    width = frames_doc.get("clamp_width_m", session.get("clamp_width_m"))

    return RunData(
        path=path,
        branch=run.branch,
        tracks=tracks,
        cols=cols,
        rows=rows,
        frames=usable,
        forces=forces,
        reference_id=reference["frame_id"],
        reference_image=read_rgb(dataset.image_path(path, reference["image_diffuse"])),
        clamp_width_m=None if width is None else float(width),
        skipped=skipped,
    )


def _usable_points(tracks: dict[int, dict], shape) -> np.ndarray:
    n = len(next(iter(tracks.values())))
    ok = np.ones(n, dtype=bool)
    for points in tracks.values():
        data = np.array([[points[i]["x"], points[i]["y"], points[i]["confidence"]] for i in range(n)])
        ok &= (
            (data[:, 2] >= CONFIDENCE)
            & (data[:, 0] >= 0)
            & (data[:, 0] < shape[1] - 1)
            & (data[:, 1] >= 0)
            & (data[:, 1] < shape[0] - 1)
        )

    return ok


def _common_cells(loading: RunData, release: RunData, H: np.ndarray) -> np.ndarray:
    cols, rows = loading.cols, loading.rows
    rr, cc = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    first = (rr * cols + cc).reshape(-1)
    ids = np.stack([first, first + 1, first + cols, first + cols + 1], axis=1)  # (cells, 4)

    xy = _xy(loading.tracks[loading.reference_id])[ids]  # (cells, 4, 2)
    target = cv2.perspectiveTransform(xy.reshape(1, -1, 2), H).reshape(xy.shape)
    grid = grid_coordinates(release.tracks[release.reference_id], cols, rows, target)
    gx, gy = grid[..., 0], grid[..., 1]
    inside = ~((grid < 0).any(axis=(1, 2)) | (gx >= cols - 1).any(axis=1) | (gy >= rows - 1).any(axis=1))

    ix = np.clip(gx.astype(int), 0, cols - 2)
    iy = np.clip(gy.astype(int), 0, rows - 2)
    corner = iy * cols + ix  # (cells, 4) top-left point of the release cell under each corner

    release_ok = _usable_points(release.tracks, release.reference_image.shape)
    touched = np.stack([corner, corner + 1, corner + cols, corner + cols + 1], axis=-1).reshape(len(ids), -1)

    loading_ok = _usable_points(loading.tracks, loading.reference_image.shape)
    valid = inside & loading_ok[ids].all(axis=1) & release_ok[touched].all(axis=1)

    return valid.reshape(rows - 1, cols - 1)


@dataclass
class Registration:
    cols: int
    rows: int
    tile: Tile
    axis: str
    homography: np.ndarray
    loading_reference: dict
    release_grid: np.ndarray  # (atlas h, w, 2) Release grid coordinates of every atlas pixel
    error_p95_px: float
    loading_id: int
    release_id: int
    vertex_grid: np.ndarray  # Release grid coordinates at the atlas cell vertices
    baseline: np.ndarray  # Loading-reference image coordinates at the same vertices

    def atlas(self) -> dict:
        t = self.tile

        return {
            "grid_cells": [t.col, t.row, t.cols, t.rows],
            "reference_bounds_px": list(t.bounds),
            "dimensions_px": list(t.size),
            "origin": "Loading reference grid top-left",
            "loading_reference_frame": self.loading_id,
            "release_reference_frame": self.release_id,
            "tracker_grid": [self.cols, self.rows],
            "active_grid_axis": self.axis,
            "loading_to_release_homography": self.homography.tolist(),
            "release_reference_error_p95_px": self.error_p95_px,
        }

    def source_map(self, run: RunData, points) -> np.ndarray:
        shape = run.reference_image.shape
        if run.branch == "loading":
            return loading_map(points, self.cols, self.tile, shape)

        return release_map(points, self.cols, self.rows, self.release_grid, shape)

    def stretch(self, run: RunData, points) -> float:
        if run.branch == "loading":
            return stretch_ratio(points, self.loading_reference, self.cols, self.tile, self.axis)

        current = sample_grid(points, self.cols, self.rows, self.vertex_grid)

        return _axis_stretch(current, self.baseline, self.axis)


def register_orientation(loading: RunData, release: RunData) -> Registration:
    if (loading.cols, loading.rows) != (release.cols, release.rows):
        raise ValueError("Loading/Release tracker topology differs; shared atlas unavailable")

    cols, rows = loading.cols, loading.rows
    H = register_references(loading.reference_image, release.reference_image)
    load_ref = loading.tracks[loading.reference_id]
    release_ref = release.tracks[release.reference_id]
    c, r, nc, nr = largest_rectangle(_common_cells(loading, release, H))
    # Keep the common grid rectangle and one output resolution for all states.
    x0, x1 = load_ref[r * cols + c]["x"], load_ref[r * cols + c + nc]["x"]
    y0, y1 = load_ref[r * cols + c]["y"], load_ref[(r + nr) * cols + c]["y"]
    tile = Tile(
        c,
        r,
        nc,
        nr,
        (x0, y0, x1, y1),
        (max(nc + 1, int(np.ceil(x1 - x0))), max(nr + 1, int(np.ceil(y1 - y0)))),
    )
    last = loading.tracks[loading.frames[-1]["frame_id"]]
    ratios = {axis: stretch_ratio(last, load_ref, cols, tile, axis) for axis in ("horizontal", "vertical")}
    axis = max(ratios, key=ratios.get)

    xx = np.linspace(0, nc, tile.size[0], dtype=np.float32)
    yy = np.linspace(0, nr, tile.size[1], dtype=np.float32)
    ci = np.minimum(xx.astype(int), nc - 1)[None, :] + c
    ri = np.minimum(yy.astype(int), nr - 1)[:, None] + r
    u = (xx[None, :] - (ci - c))[..., None]
    v = (yy[:, None] - (ri - r))[..., None]
    ids = ri * cols + ci
    ref_xy = _xy(load_ref)
    canonical = (
        (1 - u) * (1 - v) * ref_xy[ids]
        + u * (1 - v) * ref_xy[ids + 1]
        + (1 - u) * v * ref_xy[ids + cols]
        + u * v * ref_xy[ids + cols + 1]
    )
    release_xy = cv2.perspectiveTransform(canonical.reshape(1, -1, 2), H).reshape(canonical.shape)
    release_grid = grid_coordinates(release_ref, cols, rows, release_xy)
    if (
        np.any(release_grid < 0)
        or np.any(release_grid[..., 0] > cols - 1)
        or np.any(release_grid[..., 1] > rows - 1)
    ):
        raise ValueError("shared atlas leaves Release reference tracker grid")
    reference_error = np.linalg.norm(sample_grid(release_ref, cols, rows, release_grid) - release_xy, axis=-1)
    error_p95 = float(np.percentile(reference_error, 95))
    if error_p95 > MAX_REFERENCE_ERROR_PX:
        raise ValueError(f"Release atlas registration exceeds {MAX_REFERENCE_ERROR_PX:g} px reference error")

    # Release stretch is measured on the lattice of cell vertices against the Loading reference.
    vx = np.rint(np.linspace(0, tile.size[0] - 1, nc + 1)).astype(int)
    vy = np.rint(np.linspace(0, tile.size[1] - 1, nr + 1)).astype(int)

    return Registration(
        cols=cols,
        rows=rows,
        tile=tile,
        axis=axis,
        homography=H,
        loading_reference=load_ref,
        release_grid=release_grid,
        error_p95_px=error_p95,
        loading_id=loading.reference_id,
        release_id=release.reference_id,
        vertex_grid=release_grid[np.ix_(vy, vx)],
        baseline=canonical[np.ix_(vy, vx)],
    )


@dataclass(frozen=True)
class ForceTare:
    offset_n: float
    zero_counts: dict[str, float | None]  # of the reference reading of each run
    states: int = 1  # Loading states the offset was fitted to (1 = the reference reading itself)

    @property
    def zero_counts_differ(self) -> bool:
        return self.zero_counts["loading"] != self.zero_counts["release"]

    def block(self, clamp_width_m: float | None) -> dict:
        return {
            "unit": "N",
            "tare": TARE_METHOD,
            "tare_states": self.states,
            "offset_n": self.offset_n,
            "zero_counts": self.zero_counts,
            "clamp_width_m": clamp_width_m,
        }


def _reference_reading(run: RunData) -> dict:
    frame = next(f for f in run.frames if f["frame_id"] == run.reference_id)

    return dataset.force_for_frame(frame, run.forces)


def fit_tare_offset(stretch, force) -> float:
    x = np.asarray(stretch, dtype=np.float64) - 1.0
    y = np.asarray(force, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) < 1e-9:
        return float(np.median(y))

    return float(np.polyfit(x, y, 1)[1])


def measure_tare(
    loading: RunData,
    release: RunData,
    registration: Registration,
    states: int = TARE_STATES,
) -> ForceTare:
    reading = _reference_reading(loading)
    zero = {
        "loading": reading.get("zero_counts"),
        "release": _reference_reading(release).get("zero_counts"),
    }

    window = []
    for frame in loading.frames[: max(states, 1)]:
        stretch = registration.stretch(loading, loading.tracks[frame["frame_id"]])
        if stretch > TARE_MAX_STRETCH:
            break

        window.append((stretch, float(dataset.force_for_frame(frame, loading.forces)["force_n"])))

    if states <= 1 or len(window) < 3:
        return ForceTare(float(reading["force_n"]), zero, 1)

    return ForceTare(fit_tare_offset(*zip(*window, strict=True)), zero, len(window))


def tared_force(run: RunData, frame: dict, tare: ForceTare) -> tuple[float, float]:
    reading = dataset.force_for_frame(frame, run.forces)
    force = float(reading["force_n"])
    own, per_count = reading.get("zero_counts"), reading.get("calibration_n_per_count")
    reference = tare.zero_counts["loading"]
    aligned = force
    if None not in (own, per_count, reference):
        aligned += (float(reference) - float(own)) * float(per_count)

    return aligned - tare.offset_n, force


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)

    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def spearman(x, y) -> float:
    a, b = _ranks(np.asarray(x, dtype=np.float64)), _ranks(np.asarray(y, dtype=np.float64))
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def force_quality(states: list[dict]) -> dict:
    stretch = np.array([s["stretch_ratio"] for s in states])
    force = np.array([s["force_n"] for s in states])
    if len(states) < FORCE_QUALITY_MIN_STATES:
        return {"states": len(states), "usable": None}

    x = stretch - stretch.mean()
    fit = np.polyval(np.polyfit(x, force, 2), x)
    sigma = float(np.sqrt(np.sum((force - fit) ** 2) / (len(force) - 3)))
    span = float(np.ptp(fit))
    snr = span / sigma if sigma > 1e-9 else float("inf")
    rho = spearman(stretch, force)

    return {
        "states": len(states),
        "span_n": round(span, 3),
        "noise_sigma_n": round(sigma, 3),
        "snr": round(snr, 2) if np.isfinite(snr) else None,
        "spearman": round(rho, 3),
        "usable": bool(snr >= FORCE_MIN_SNR and rho >= FORCE_MIN_SPEARMAN),
    }


@dataclass(frozen=True)
class PsSetup:
    reference: ps_frames.Reference
    albedo_scale: float
    states: int  # number of states the scale was measured over

    @property
    def light(self) -> ph.LightConfig:
        return self.reference.light

    def block(self, tilts: list[float]) -> dict:
        ref = self.reference

        return {
            "albedo_scale": self.albedo_scale,
            "albedo_scale_basis": albedo_scale_basis(self.states),
            "albedo_encoding": ALBEDO_ENCODING,
            "exposure_gain": ref.exposure_gain,
            "exposure_basis": EXPOSURE_BASIS,
            "normal_encoding": NORMAL_ENCODING,
            "normal_correction": NORMAL_CORRECTION,
            "light": {
                "azimuths_deg": list(self.light.azimuths_deg),
                "elevation_deg": self.light.elevation_deg,
                "source": self.light.source,
            },
            "gains": list(self.light.gains),
            "gains_mode": "per-state" if ref.track_gains else "fixed",
            "quality": {
                "albedo_diffuse_correlation": ref.quality["albedo_diffuse_correlation"],
                "normal_tilt_deg": {
                    "mean": round(float(np.mean(tilts)), 2),
                    "max": round(float(np.max(tilts)), 2),
                },
                "warnings": ref.warnings,
            },
            "frame": PS_FRAME,
        }


class PsAtlas(NamedTuple):
    albedo: np.ndarray  # float RGB, multiplied by the exposure gain
    normals: np.ndarray | None  # float unit vectors, tilt removed; None when not requested
    gains: tuple[float, float, float]
    tilt_deg: float | None  # tilt of the mean normal before the correction


def _ps_problem(run: RunData, frame: dict) -> str | None:
    for key in dataset.PS_KEYS:
        problem = dataset.image_problem(dataset.image_path(run.path, frame[key]))
        if problem:
            return f"{frame[key]}: {problem}"

    return None


def _ps_size_differs(run: RunData, frame: dict) -> bool:
    sizes = set()
    for key in ("image_diffuse", *dataset.PS_KEYS):
        with Image.open(dataset.image_path(run.path, frame[key])) as image:
            sizes.add(image.size)

    return len(sizes) != 1


def prepare_ps(
    loading: RunData,
    release: RunData,
    led_config: Path | None,
    track_gains: bool = True,
) -> tuple[ps_frames.Reference, list[tuple[RunData, dict, str]]]:
    damaged = []
    for run in (loading, release):
        unlisted = [f["frame_id"] for f in run.frames if not dataset.has_ps(f)]
        if unlisted:
            raise ValueError(f"{run.path.name}: {len(unlisted)} state(s) without photometric-stereo images")

        intact = []
        for frame in run.frames:
            problem = _ps_problem(run, frame)
            if problem is None:
                intact.append(frame)
            elif frame["frame_id"] == run.reference_id:
                raise ValueError(f"{run.path.name}: reference frame {frame['frame_id']}: {problem}")
            else:
                damaged.append((run, frame, problem))

        mismatched = [f["frame_id"] for f in intact if _ps_size_differs(run, f)]
        if mismatched:
            raise ValueError(
                f"{run.path.name}: {len(mismatched)} state(s) whose photometric-stereo images differ "
                "in size from the diffuse image"
            )

    cfg = ph.load_light_config(led_config, session=dataset.load_session(loading.path))
    reference = ps_frames.measure_reference(loading.path, dataset.load_frames(loading.path), cfg)

    if not track_gains:
        reference = replace(reference, track_gains=False)

    return reference, damaged


def ps_atlas(
    run: RunData,
    frame: dict,
    reference: ps_frames.Reference,
    mapping: np.ndarray,
    normals: bool = True,
) -> PsAtlas:
    ps = ph.read_ps_gray([dataset.image_path(run.path, frame[k]) for k in dataset.PS_KEYS])

    diffuse = ph.read_rgb_linear(dataset.image_path(run.path, frame["image_diffuse"]))
    if diffuse.shape[:2] != ps.shape[:2]:
        raise ValueError("diffuse and PS images differ in size")

    light = ps_frames.frame_light(reference, ps)
    albedo_gray, normal = ph.solve(ps, light)
    albedo = warp(ph.color_albedo(albedo_gray, diffuse) * reference.exposure_gain, mapping)

    gains = tuple(float(g) for g in light.gains)
    if not normals:
        return PsAtlas(albedo, None, gains, None)

    normal = warp(normal, mapping)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.where(length > 1e-6, normal / np.maximum(length, 1e-6), np.float32([0.0, 0.0, 1.0]))

    normal, tilt = ph.remove_normal_tilt(normal)

    return PsAtlas(albedo, normal, gains, tilt)


def clip_threshold(albedo: np.ndarray, fraction: float = 0.001) -> float:
    peak = np.asarray(albedo, dtype=np.float32).max(axis=-1).reshape(-1)
    kth = peak.size - 1 - int(peak.size * fraction)
    value = float(np.partition(peak, kth)[kth])

    return value if np.isfinite(value) and value > 1e-6 else 1.0


def measure_albedo_scale(
    jobs: list[tuple[RunData, dict, dict]],
    reference: ps_frames.Reference,
    registration: Registration,
    seams=None,
) -> float:
    scale = 0.0
    for run, frame, state in jobs:
        mapping = registration.source_map(run, run.tracks[state["frame_id"]])
        albedo = ps_atlas(run, frame, reference, mapping, normals=False).albedo
        peak = clip_threshold(albedo)
        if seams is not None:
            from . import tiling as tiling_mod

            wide = 2 * peak  # headroom, so that the tile is measured before anything clips

            tile = tiling_mod.apply_tiling(ph.encode_albedo_u16(albedo, wide), seams)
            peak = max(peak, clip_threshold(ph.decode_albedo_u16(tile, wide)))

        scale = max(scale, peak)

    return scale * ALBEDO_HEADROOM


def clipped_fraction(albedo_u16: np.ndarray) -> float:
    return float(
        np.count_nonzero((albedo_u16 == 65535).any(axis=-1)) / (albedo_u16.shape[0] * albedo_u16.shape[1])
    )


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), _PNG_PARAMS):
        raise OSError(f"cannot write {path}")


def _state_name(branch: str, kind: str, frame_id: int) -> str:
    return f"{branch}/{kind}_{frame_id:04d}.png"


def _clear(*paths: Path | None) -> None:
    for path in paths:
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)


def _pick_states(branches: dict[str, list[dict]]) -> dict[str, int]:
    picks, offset = {}, 0
    for branch, states in branches.items():
        order = sorted(range(len(states)), key=lambda i: states[i]["stretch_ratio"])
        picks[f"{branch} low"] = offset + order[0]
        picks[f"{branch} mid"] = offset + order[len(order) // 2]
        picks[f"{branch} high"] = offset + order[-1]
        offset += len(states)

    return picks


def build_orientation(
    orientation: str,
    runs: dict[str, dataset.Run],
    material: str,
    untiled: Path | None,
    tiled: Path | None,
    tiling=None,
    ps: bool = True,
    led_config: Path | None = None,
    diagnostics: Path | None = None,
    tare_states: int = TARE_STATES,
    ps_gains: str = "per-state",
) -> tuple[dict, dict[Path, dict]]:
    loading, release = load_run(runs["loading"]), load_run(runs["release"])
    registration = register_orientation(loading, release)
    report: dict = {"status": "built", "atlas": registration.atlas()}
    warnings: list[str] = []
    reference = None
    if ps:
        try:
            reference, damaged = prepare_ps(loading, release, led_config, ps_gains == "per-state")
        except (ValueError, OSError) as exc:
            report["ps_skipped"] = str(exc)
        else:
            # layers are all-or-nothing in a package: a state with unusable PS images is left out
            for run, frame, _ in damaged:
                run.frames = [f for f in run.frames if f["frame_id"] != frame["frame_id"]]
            if damaged:
                report["dropped_states"] = [
                    {"branch": run.branch, "frame_id": frame["frame_id"], "reason": why}
                    for run, frame, why in damaged
                ]
                warnings += [
                    f"{run.branch} frame {frame['frame_id']} left out: photometric-stereo image {why}"
                    for run, frame, why in damaged
                ]
            warnings += reference.warnings
    tare = measure_tare(loading, release, registration, tare_states)
    report["skipped_states"] = loading.skipped + release.skipped
    if loading.clamp_width_m != release.clamp_width_m:
        report["clamp_width_warning"] = (
            f"loading {loading.clamp_width_m} m, release {release.clamp_width_m} m"
        )

    # Pass 1: RGB states.  Untiled images are written as they are made; tiling needs all of them.
    branches: dict[str, list[dict]] = {}
    jobs: list[tuple[RunData, dict, dict]] = []
    textures = []
    for run in (loading, release):
        states = []
        for frame in run.frames:
            fid = frame["frame_id"]
            points = run.tracks[fid]
            texture = warp(
                read_rgb(dataset.image_path(run.path, frame["image_diffuse"])),
                registration.source_map(run, points),
            )
            force_n, force_raw_n = tared_force(run, frame, tare)
            state = {
                "frame_id": fid,
                "sequence_index": len(states),
                "stretch_ratio": registration.stretch(run, points),
                "force_n": force_n,
                "force_raw_n": force_raw_n,
                "travel_mm": frame.get("displacement_mm"),
                "texture": _state_name(run.branch, "frame", fid),
                "source_image": frame["image_diffuse"],
            }
            if reference is not None:
                state["albedo"] = _state_name(run.branch, "albedo", fid)
                state["normal"] = _state_name(run.branch, "normal", fid)
            states.append(state)
            jobs.append((run, frame, state))
            if tiling is not None and tiled is not None:
                textures.append(texture)
            if untiled is not None:
                write_png(untiled / state["texture"], texture)
        branches[run.branch] = states
    residual = next(s for s in branches["release"] if s["frame_id"] == release.reference_id)["stretch_ratio"]
    report["states"] = {b: len(s) for b, s in branches.items()}
    report["stretch_range"] = {
        b: [
            min(s["stretch_ratio"] for s in states),
            max(s["stretch_ratio"] for s in states),
        ]
        for b, states in branches.items()
    }
    report["release_residual_stretch"] = residual
    if residual < 1.0:
        warnings.append(
            f"release residual stretch {residual:.4f} is below 1: the cloth ends shorter than its loading "
            "reference, which is within the noise of the stretch measurement"
        )
    quality = {b: force_quality(states) for b, states in branches.items()}
    for branch, figures in quality.items():
        if figures["usable"] is False:
            warnings.append(
                f"{branch} force is not usable: span {figures['span_n']} N against noise "
                f"{figures['noise_sigma_n']} N (ratio {figures['snr']}, rank correlation with stretch "
                f"{figures['spearman']}); do not drive a view or a fit with it"
            )
    report["force"] = {
        "offset_n": tare.offset_n,
        "tare_states": tare.states,
        "quality": quality,
        "zero_counts": tare.zero_counts,
        "zero_counts_differ": tare.zero_counts_differ,
        "range_n": {
            b: [min(s["force_n"] for s in states), max(s["force_n"] for s in states)]
            for b, states in branches.items()
        },
    }

    # Tiling: one solution for all RGB states, kept as seams for the photometric layers.
    seams = tiling_meta = None
    if textures:
        from . import tiling as tiling_mod

        started = time.perf_counter()
        try:
            tiles, tiling_meta, details = tiling_mod.tile_states(textures, tiling)
        except ValueError as exc:  # parameters do not fit this atlas: the orientation stays untiled
            report["tiling_skipped"] = str(exc)
        else:
            for (_, _, state), tile in zip(jobs, tiles, strict=True):
                write_png(tiled / state["texture"], tile)
            seams = tiling_mod.Seams.from_details(details, registration.tile.size)
            seams.save(tiled / SEAMS_FILE)
            if diagnostics is not None:
                tiling_mod.write_diagnostics(
                    diagnostics,
                    orientation,
                    textures,
                    tiles,
                    tiling_meta,
                    details,
                    _pick_states(branches),
                )
            tiling_meta["quality"] = tiling_mod.quality_checks(tiles, textures)
            tiling_meta["total_seconds"] = round(time.perf_counter() - started, 2)
            report["tiling"] = tiling_meta
        del textures

    # Pass 2: photometric layers, one state at a time (untiled and tiled).  The albedo scale needs
    # every state first, so the albedo is solved twice: once to measure, once to encode.
    setup = None
    if reference is not None:
        setup = PsSetup(
            reference,
            measure_albedo_scale(jobs, reference, registration, seams),
            len(jobs),
        )
        clipped = {"untiled": 0.0}
        tilts = []
        for run, frame, state in jobs:
            mapping = registration.source_map(run, run.tracks[state["frame_id"]])
            layers = ps_atlas(run, frame, setup.reference, mapping)
            state["ps_gains"] = [round(g, 5) for g in layers.gains]
            state["normal_tilt_deg"] = round(layers.tilt_deg, 2)
            tilts.append(layers.tilt_deg)
            albedo = ph.encode_albedo_u16(layers.albedo, setup.albedo_scale)
            normal = ph.encode_normals_u16(layers.normals)
            clipped["untiled"] = max(clipped["untiled"], clipped_fraction(albedo))
            if untiled is not None:
                write_png(untiled / state["albedo"], albedo)
                write_png(untiled / state["normal"], normal)
            if seams is not None:
                from . import tiling as tiling_mod

                albedo_tiled = tiling_mod.apply_tiling(albedo, seams)
                clipped["tiled"] = max(clipped.get("tiled", 0.0), clipped_fraction(albedo_tiled))
                write_png(tiled / state["albedo"], albedo_tiled)
                write_png(
                    tiled / state["normal"],
                    tiling_mod.apply_tiling(normal, seams, normals=True),
                )
        block = setup.block(tilts)
        report["ps"] = {
            "albedo_scale": setup.albedo_scale,
            "exposure_gain": block["exposure_gain"],
            "gains": block["gains"],
            "gains_mode": block["gains_mode"],
            "quality": block["quality"],
            "clipped_fraction_max": clipped,
        }

    metadata = {
        "format": FORMAT,
        "version": VERSION,
        "material": material,
        "orientation": orientation,
        "atlas": registration.atlas(),
        "lref_mm": None,
        "lref_status": "unknown: physical reference length not measured",
        "stretch_definition": STRETCH_DEFINITION,
        "release_residual_stretch": residual,
        "color_space": "sRGB PNG; decode to linear RGB before interpolation",
        "force": tare.block(loading.clamp_width_m),
    }
    metadata["force"]["quality"] = quality
    if setup is not None:
        metadata["ps"] = block

    metadata["branches"] = branches
    report["warnings"] = warnings
    written = {}
    if untiled is not None:
        written[untiled] = metadata

    if seams is not None:
        written[tiled] = {**metadata, "tiling": tiling_meta}

    return report, written


def build(
    project: Path,
    tiled_out: Path | None = None,
    untiled_out: Path | None = None,
    tiling=None,
    *,
    ps: bool = True,
    orientations: list[str] | None = None,
    led_config: Path | None = None,
    diagnostics_out: Path | None = None,
    force: bool = False,
    tare_states: int = TARE_STATES,
    ps_gains: str = "per-state",
) -> dict:
    if ps_gains not in ("per-state", "reference"):
        raise ValueError(f"ps_gains must be 'per-state' or 'reference', not {ps_gains!r}")
    if tiled_out is not None and tiling is None:
        raise ValueError("tiled_out needs a tiling configuration")
    outputs = [p for p in (tiled_out, untiled_out) if p is not None]
    if not outputs:
        raise ValueError("no output directory")
    for path in outputs:
        if path.exists() and any(path.iterdir()) and not force:
            raise FileExistsError(f"{path} exists and is not empty (use --force to overwrite)")
    grouped = dataset.discover_orientations(project, orientations)
    report: dict = {
        "project": project.name,
        "format": FORMAT,
        "version": VERSION,
        "orientations": {},
    }
    ignored = dataset.ignored_run_folders(project)
    if ignored:
        report["ignored_run_folders"] = [{"name": name, "reason": reason} for name, reason in ignored]
    for name in orientations or []:
        if name not in grouped:
            report["orientations"][name] = {
                "status": "skipped",
                "reason": "no run folders found",
            }
    for orientation, runs in grouped.items():
        untiled = None if untiled_out is None else untiled_out / orientation
        tiled = None if tiled_out is None else tiled_out / orientation
        diagnostics = None if diagnostics_out is None else diagnostics_out / orientation
        _clear(untiled, tiled, diagnostics)  # an earlier build must not outlive a failure now
        missing = [b for b in dataset.BRANCHES if b not in runs]
        if missing:
            report["orientations"][orientation] = {
                "status": "skipped",
                "reason": f"no {' and '.join(missing)} run (both loading and release are needed)",
            }
            continue
        started = time.perf_counter()
        try:
            entry, written = build_orientation(
                orientation,
                runs,
                project.name,
                untiled,
                tiled,
                tiling,
                ps,
                led_config,
                diagnostics,
                tare_states,
                ps_gains,
            )
            for directory, metadata in written.items():
                (directory / "package.json").write_text(json.dumps(metadata, indent=2) + "\n")
        except (OSError, KeyError, ValueError) as exc:
            _clear(untiled, tiled, diagnostics)
            report["orientations"][orientation] = {
                "status": "skipped",
                "reason": str(exc) or type(exc).__name__,
            }
            continue
        except BaseException:
            _clear(untiled, tiled, diagnostics)
            raise
        entry["seconds"] = round(time.perf_counter() - started, 1)
        report["orientations"][orientation] = entry
    for root in outputs:
        root.mkdir(parents=True, exist_ok=True)
        (root / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")

    return report


def _tiling_config(args: argparse.Namespace):
    from .tiling import TilingConfig

    def one(values):
        return values[0] if len(values) == 1 else list(values)

    return TilingConfig(
        tuple(args.tiling_window_size),
        one(args.tiling_shift),
        one(args.tiling_interp_dist),
        args.tiling_guidance,
        args.tiling_area,
        args.tiling_joint_search,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fabsim package",
        description="Build a material package from tracked measurement runs.",
    )
    parser.add_argument(
        "project",
        type=Path,
        help="project folder with <orientation>_loading/_release runs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="tiled package (default: <project>/material_package_v3)",
    )
    parser.add_argument(
        "--untiled-out",
        type=Path,
        help="untiled package (default: <project>/material_package_v3_untiled)",
    )
    parser.add_argument("--no-tiling", action="store_true", help="write only the untiled package")
    parser.add_argument("--no-ps", action="store_true", help="skip albedo and normal layers")
    parser.add_argument(
        "--ps-gains",
        choices=("per-state", "reference"),
        default="per-state",
        help="estimate the LED gains for every state (default) or once on the Loading reference frame",
    )
    parser.add_argument(
        "--tare-states",
        type=int,
        default=TARE_STATES,
        metavar="N",
        help="fit the force tare to the first N Loading states (default %(default)s; 1 = use the "
        "Loading reference reading alone)",
    )
    parser.add_argument(
        "--orientations",
        nargs="+",
        metavar="NAME",
        help="only these (default: all found)",
    )
    parser.add_argument(
        "--led-config",
        type=Path,
        help="LED configuration JSON for runs without an led_config snapshot in session.json "
        "(the snapshot wins; default: packaged configuration)",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        metavar="DIR",
        help="also write tiling previews to DIR/<orientation>/",
    )
    parser.add_argument("--force", action="store_true", help="overwrite non-empty output folders")
    tiling = parser.add_argument_group("tiling (MATLAB findTile port, see docs/texture_tiling.md)")
    tiling.add_argument(
        "--tiling-window-size",
        type=int,
        nargs="+",
        default=[100, 200],
        metavar="N",
        help="[vertical horizontal] or one value (default 100 200)",
    )
    tiling.add_argument(
        "--tiling-shift",
        type=int,
        nargs="+",
        default=[20],
        metavar="N",
        help="one value or left right top down (default 20)",
    )
    tiling.add_argument(
        "--tiling-interp-dist",
        type=int,
        nargs="+",
        default=[2],
        metavar="N",
        help="one value or vertical horizontal (default 2)",
    )
    tiling.add_argument(
        "--tiling-guidance",
        type=int,
        nargs="+",
        metavar="STATE",
        help="state indices (Loading then Release) that guide the seam search (default: all)",
    )
    tiling.add_argument(
        "--tiling-area",
        type=int,
        nargs="+",
        metavar="N",
        help="tile area in the atlas, 0-based [top left height width] (default: automatic)",
    )
    tiling.add_argument(
        "--tiling-joint-search",
        action="store_true",
        help="score bottom and right edges together",
    )
    args = parser.parse_args(argv)
    if not args.project.is_dir():
        parser.error(f"{args.project} is not a directory")
    if args.tiling_guidance and min(args.tiling_guidance) < 0:
        parser.error("--tiling-guidance: state indices must not be negative")
    tiled = None if args.no_tiling else (args.out or args.project / "material_package_v3")
    untiled = args.untiled_out or args.project / "material_package_v3_untiled"
    try:
        report = build(
            args.project,
            tiled,
            untiled,
            None if args.no_tiling else _tiling_config(args),
            ps=not args.no_ps,
            orientations=args.orientations,
            led_config=args.led_config,
            diagnostics_out=args.diagnostics,
            force=args.force,
            tare_states=max(1, args.tare_states),
            ps_gains=args.ps_gains,
        )
    except FileExistsError as exc:
        parser.error(str(exc))
    for item in report.get("ignored_run_folders", []):
        print(
            f"warning: {item['name']}: not used as a run ({item['reason']})",
            file=sys.stderr,
        )
    built = 0
    for name, entry in report["orientations"].items():
        if entry["status"] != "built":
            print(f"{name}: skipped ({entry['reason']})")
            continue
        built += 1
        notes = [k for k in ("ps_skipped", "tiling_skipped") if k in entry]
        counts = ", ".join(f"{n} {b}" for b, n in entry["states"].items())
        print(f"{name}: {counts} states, release residual stretch {entry['release_residual_stretch']:.4f}")
        for key in notes:
            print(f"  {key.replace('_', ' ')}: {entry[key]}")
        for text in entry["warnings"]:
            print(f"  warning: {text}")
    print(f"reports: {', '.join(str(p / 'build_report.json') for p in (tiled, untiled) if p)}")

    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
