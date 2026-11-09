from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# Bumped whenever the adapter's behaviour or output schema changes
ADAPTER_VERSION = "1.1.0"

# Environment variables the front end (``server/fabsim-track``) sets so the
# provenance can name things this file cannot discover by itself.
ENV_ADAPTER_COMMIT = "FABSIM_ADAPTER_COMMIT"
ENV_RUN_KEY = "FABSIM_RUN_KEY"
ENV_CHECKPOINT_SHA256 = "COWTRACKER_CHECKPOINT_SHA256"

# Upstream model/API identity this adapter was implemented against.
COWTRACKER_REPO = "https://github.com/facebookresearch/cowtracker"
COWTRACKER_COMMIT = "1454f20045d3b514e5b8417907152677f3dba621"
COWTRACKER_MODEL_REPO = "facebook/cowtracker"
COWTRACKER_MODEL_FILENAME = "cowtracker_model.pth"

FABSIM_TRACKS_FORMAT = "fabsim-tracks"
FABSIM_TRACKS_VERSION = 1

# Upstream demo defaults (H, W) = (336, 560).
DEFAULT_INFERENCE_WIDTH = 560
DEFAULT_INFERENCE_HEIGHT = 336
DEFAULT_REFERENCE_ERROR_LIMIT = 2.0

PIXEL_CENTER = "pixel_center"
CORNER_ALIGNED = "corner_aligned"
CONVENTIONS = (PIXEL_CENTER, CORNER_ALIGNED)

CONFIDENCE_MODES = ("vis*conf", "vis", "conf", "min")

DIAGNOSTIC_PERCENTS = (0, 25, 50, 75, 100)  # first / 25 / 50 / 75 / last


class AdapterError(Exception):
    pass


@dataclass
class TrackPoint:
    id: str
    x: float
    y: float


@dataclass
class SequenceFrame:
    sequence_index: int
    frame_id: int
    image_path: str


@dataclass
class RunPlan:
    run_dir: str
    role: str
    frames: list[SequenceFrame]
    skipped: list[dict[str, Any]]
    source_w: int | None
    source_h: int | None
    inference_w: int
    inference_h: int
    scale_x: float | None
    scale_y: float | None
    points: list[TrackPoint]
    points_mode: str  # "points-file" | "grid"
    reference_frame_id: int | None
    reference_seq_index: int | None
    rotation_active: bool
    convention: str
    confidence_mode: str
    reference_error_limit: float
    output_path: str
    csv_output: str | None
    save_sampled: str | None
    save_dense: str | None
    diagnostics_dir: str | None
    diagnostic_video: str | None
    device: str
    notes: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def source_to_inference(
    x: float,
    y: float,
    src_w: int,
    src_h: int,
    inf_w: int,
    inf_h: int,
    convention: str = PIXEL_CENTER,
) -> tuple[float, float]:
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown coordinate convention {convention!r}")

    if src_w <= 0 or src_h <= 0 or inf_w <= 0 or inf_h <= 0:
        raise ValueError("dimensions must be positive")

    if convention == PIXEL_CENTER:
        xi = (x + 0.5) * (inf_w / src_w) - 0.5
        yi = (y + 0.5) * (inf_h / src_h) - 0.5
    else:
        xi = x * (inf_w / src_w)
        yi = y * (inf_h / src_h)

    return xi, yi


def inference_to_source(
    x: float,
    y: float,
    src_w: int,
    src_h: int,
    inf_w: int,
    inf_h: int,
    convention: str = PIXEL_CENTER,
) -> tuple[float, float]:
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown coordinate convention {convention!r}")

    if src_w <= 0 or src_h <= 0 or inf_w <= 0 or inf_h <= 0:
        raise ValueError("dimensions must be positive")

    if convention == PIXEL_CENTER:
        xs = (x + 0.5) * (src_w / inf_w) - 0.5
        ys = (y + 0.5) * (src_h / inf_h) - 0.5
    else:
        xs = x * (src_w / inf_w)
        ys = y * (src_h / inf_h)

    return xs, ys


def build_sequence(
    run_dir: str, role: str, skip_missing: bool
) -> tuple[list[SequenceFrame], list[dict[str, Any]]]:
    frames_path = os.path.join(run_dir, "frames.json")
    if not os.path.isfile(frames_path):
        raise AdapterError(f"run directory has no frames.json: {frames_path}")

    with open(frames_path) as fh:
        try:
            doc = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{frames_path}: invalid JSON: {exc}") from exc

    if not isinstance(doc, dict) or not isinstance(doc.get("frames"), list):
        raise AdapterError(f'{frames_path}: missing "frames" array')

    images_dir = os.path.join(run_dir, "images")
    key = f"image_{role}"
    frames: list[SequenceFrame] = []
    skipped: list[dict[str, Any]] = []
    for seq_index, frame_entry in enumerate(doc["frames"]):
        if not isinstance(frame_entry, dict):
            skipped.append(
                {
                    "sequence_index": seq_index,
                    "frame_id": None,
                    "reason": "frame entry is not an object",
                }
            )
            continue

        frame_id = frame_entry.get("frame_id")
        name = frame_entry.get(key)
        if not isinstance(name, str) or not name:
            skipped.append(
                {
                    "sequence_index": seq_index,
                    "frame_id": frame_id,
                    "reason": f'no "{key}" entry in frames.json',
                }
            )
            continue

        candidate = os.path.join(images_dir, name)
        if not os.path.isfile(candidate):
            fallback = os.path.join(run_dir, name)
            if not os.path.isfile(fallback):
                skipped.append(
                    {
                        "sequence_index": seq_index,
                        "frame_id": frame_id,
                        "reason": f'{key} "{name}" not found on disk',
                    }
                )
                continue

            candidate = fallback
        if not isinstance(frame_id, int) or isinstance(frame_id, bool):
            raise AdapterError(
                f"{frames_path}: frame at sequence index {seq_index} has no "
                f"numeric frame_id (got {frame_id!r})"
            )

        frames.append(
            SequenceFrame(
                sequence_index=seq_index, frame_id=frame_id, image_path=candidate
            )
        )

    if skipped and not skip_missing:
        detail = "\n".join(
            f"  sequence_index {s['sequence_index']} frame_id {s['frame_id']}: {s['reason']}"
            for s in skipped
        )

        raise AdapterError(
            f"{len(skipped)} frame(s) lack the requested role {role!r} "
            f'(missing "{key}" or file); rerun with --skip-missing to skip them.\n{detail}'
        )

    if not frames:
        raise AdapterError(f"no frames with role {role!r} after filtering")

    return frames, skipped


def probe_image(path: str) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size

    except Exception as exc:  # PIL raises several concrete types
        raise AdapterError(f"cannot read image dimensions of {path}: {exc}") from exc


def probe_sequence_dimensions(frames: list[SequenceFrame]) -> tuple[int, int]:
    dims = [probe_image(f.image_path) for f in frames]
    first = dims[0]
    for f, d in zip(frames, dims, strict=False):
        if d != first:
            raise AdapterError(
                f"inconsistent source image dimensions: frame_id {f.frame_id} "
                f"is {d[0]}x{d[1]} but the first frame is {first[0]}x{first[1]}"
            )

    return first


def decode_image(path: str) -> np.ndarray:
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:
        raise AdapterError(f"cannot decode image {path}: {exc}") from exc

    return arr


def pixel_center_resize(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    arr = np.asarray(image)
    single = arr.ndim == 2
    if single:
        arr = arr[:, :, None]

    h, w = arr.shape[0], arr.shape[1]
    if (h, w) == (out_h, out_w):
        out = arr
    else:
        x_in = np.clip(
            (np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5,
            0.0,
            float(w - 1),
        )
        y_in = np.clip(
            (np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5,
            0.0,
            float(h - 1),
        )
        x0 = np.floor(x_in).astype(np.int64)
        y0 = np.floor(y_in).astype(np.int64)
        wx = (x_in - x0).astype(np.float32)
        wy = (y_in - y0).astype(np.float32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        row0 = arr[y0][:, x0]  # [out_h, out_w, C]
        row1 = arr[y1][:, x0]
        row2 = arr[y0][:, x1]
        row3 = arr[y1][:, x1]
        out = (1 - wy)[:, None, None] * (
            (1 - wx)[None, :, None] * row0 + wx[None, :, None] * row2
        ) + wy[:, None, None] * (
            (1 - wx)[None, :, None] * row1 + wx[None, :, None] * row3
        )
    if single:
        out = out[:, :, 0]

    return out


def load_points(path: str) -> list[TrackPoint]:
    if not os.path.isfile(path):
        raise AdapterError(f"points file not found: {path}")
    with open(path) as fh:
        try:
            doc = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{path}: invalid JSON: {exc}") from exc

    raw_points: list[tuple[Any, Any, Any]] = []
    if isinstance(doc, dict) and "points" in doc:
        points_value = doc["points"]
        if isinstance(points_value, list):
            for entry in points_value:
                if not isinstance(entry, dict):
                    raise AdapterError(
                        f"{path}: point entry is not an object: {entry!r}"
                    )
                pid = entry.get("id")
                if pid is None:
                    raise AdapterError(f'{path}: point entry missing "id": {entry!r}')
                raw_points.append((pid, entry.get("x"), entry.get("y")))
        elif isinstance(points_value, dict):
            for pid, entry in points_value.items():
                if not isinstance(entry, dict):
                    raise AdapterError(f"{path}: point {pid!r} is not an object")
                raw_points.append((pid, entry.get("x"), entry.get("y")))
        else:
            raise AdapterError(f'{path}: "points" must be an array or object')
    elif isinstance(doc, list):
        for entry in doc:
            if not isinstance(entry, dict):
                raise AdapterError(f"{path}: point entry is not an object: {entry!r}")
            pid = entry.get("id")
            if pid is None:
                raise AdapterError(f'{path}: point entry missing "id": {entry!r}')
            raw_points.append((pid, entry.get("x"), entry.get("y")))
    else:
        raise AdapterError(f'{path}: expected {{"points": [...]}} or a bare array')

    points: list[TrackPoint] = []
    seen: dict[str, int] = {}
    for pid, x, y in raw_points:
        sid = str(pid)
        if sid in seen:
            raise AdapterError(f"{path}: duplicate point id {sid!r}")
        seen[sid] = len(points)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise AdapterError(f"{path}: point {sid!r} has non-numeric x/y")
        xf, yf = float(x), float(y)
        if not (np.isfinite(xf) and np.isfinite(yf)):
            raise AdapterError(f"{path}: point {sid!r} has non-finite coordinates")
        points.append(TrackPoint(id=sid, x=xf, y=yf))
    if not points:
        raise AdapterError(f"{path}: no points found")

    return points


def grid_points(
    cols: int,
    rows: int,
    src_w: int,
    src_h: int,
    roi: tuple[float, float, float, float] | None = None,
    spacing: tuple[float, float] | None = None,
) -> list[TrackPoint]:
    if (cols < 1 or rows < 1) and spacing is None:
        raise AdapterError("--grid-cols and --grid-rows must be >= 1")
    if roi is None:
        rx, ry, rw, rh = 0.0, 0.0, float(src_w), float(src_h)
    else:
        rx, ry, rw, rh = (float(v) for v in roi)
        if rw <= 0 or rh <= 0:
            raise AdapterError("--grid-roi must have positive width and height")
    if spacing is not None:
        sx, sy = (float(v) for v in spacing)
        if sx <= 0 or sy <= 0:
            raise AdapterError("--grid-spacing must be positive")
        xs = []
        x = rx
        while x < rx + rw - 1e-9:
            xs.append(x)
            x += sx
        ys = []
        y = ry
        while y < ry + rh - 1e-9:
            ys.append(y)
            y += sy
    else:
        xs = [rx - 0.5 + (i + 0.5) * rw / cols for i in range(cols)]
        ys = [ry - 0.5 + (j + 0.5) * rh / rows for j in range(rows)]
    points: list[TrackPoint] = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            k = j * len(xs) + i
            points.append(TrackPoint(id=f"grid_{k}", x=x, y=y))

    return points


def validate_points_in_image(
    points: list[TrackPoint],
    src_w: int,
    src_h: int,
    inf_w: int,
    inf_h: int,
    convention: str,
) -> list[str]:
    notes: list[str] = []
    if convention == PIXEL_CENTER:
        in_bounds = lambda v, size: -0.5 <= v <= size - 0.5  # noqa: E731
    else:
        in_bounds = lambda v, size: 0.0 <= v <= size  # noqa: E731
    for p in points:
        if not (in_bounds(p.x, src_w) and in_bounds(p.y, src_h)):
            notes.append(
                f"point {p.id!r} ({p.x:.3f}, {p.y:.3f}) lies outside the "
                f"source image {src_w}x{src_h}; sampling will clamp to the field edge"
            )
            continue
        xi, yi = source_to_inference(p.x, p.y, src_w, src_h, inf_w, inf_h, convention)
        if not (0.0 <= xi <= inf_w - 1 and 0.0 <= yi <= inf_h - 1):
            notes.append(
                f"point {p.id!r} ({p.x:.3f}, {p.y:.3f}) maps outside the model "
                f"inference field {inf_w}x{inf_h} (inference ({xi:.3f}, {yi:.3f})); "
                "the reference reconstruction error check will reject it"
            )

    return notes


def sample_dense_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    field = np.asarray(field)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if field.ndim == 4:
        T, H, W, C = field.shape
        squeezed = False
    elif field.ndim == 3:
        T, H, W = field.shape
        field = field[:, :, :, None]
        squeezed = True
    else:
        raise ValueError(f"field must be [T,H,W,C] or [T,H,W], got {field.shape}")
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")
    xs = np.clip(xs, 0.0, float(W - 1))
    ys = np.clip(ys, 0.0, float(H - 1))
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)
    f00 = field[:, y0, x0]
    f01 = field[:, y0, x1]
    f10 = field[:, y1, x0]
    f11 = field[:, y1, x1]
    out = (1 - wy)[None, :, None] * (
        (1 - wx)[None, :, None] * f00 + wx[None, :, None] * f01
    ) + wy[None, :, None] * ((1 - wx)[None, :, None] * f10 + wx[None, :, None] * f11)
    if squeezed:
        out = out[:, :, 0]

    return out


def combine_confidence(
    visibility: np.ndarray, raw_confidence: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "vis*conf":
        return visibility * raw_confidence
    if mode == "vis":
        return visibility
    if mode == "conf":
        return raw_confidence
    if mode == "min":
        return np.minimum(visibility, raw_confidence)
    raise ValueError(f"unknown confidence mode {mode!r}")


def compute_rotation(ref_seq_index: int, frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.int64)

    return (
        (np.arange(frame_count, dtype=np.int64) - ref_seq_index) % frame_count
    ).astype(np.int64)


def reference_error_px(
    track_src: np.ndarray, query_src: np.ndarray, ref_seq_index: int
) -> np.ndarray:
    d = track_src[ref_seq_index, :, :] - query_src

    return np.hypot(d[:, 0], d[:, 1])


def collect_runtime_info(torch_mod: Any, device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "torch": torch_mod.__version__,
        "cuda": getattr(torch_mod.version, "cuda", None),
        "device": device,
    }
    if device == "cuda":
        info["cudnn"] = torch_mod.backends.cudnn.version()
        info["gpu"] = torch_mod.cuda.get_device_name(0)

    return info


def checkpoint_identity(args: argparse.Namespace) -> dict[str, Any]:
    ident: dict[str, Any] = {
        "checkpoint": (
            os.path.basename(args.checkpoint)
            if args.checkpoint
            else f"huggingface:{COWTRACKER_MODEL_REPO}/{COWTRACKER_MODEL_FILENAME}"
        )
    }
    sha = os.environ.get(ENV_CHECKPOINT_SHA256)
    if sha:
        ident["checkpoint_sha256"] = sha

    return ident


def run_label(run_dir: str) -> str:
    key = os.environ.get(ENV_RUN_KEY)
    if key:
        return key

    return "/".join(os.path.abspath(run_dir).rstrip(os.sep).split(os.sep)[-2:])


def build_provenance(
    plan: RunPlan,
    args: argparse.Namespace,
    reference_errors: np.ndarray,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prov: dict[str, Any] = {
        "tool": "cowtracker_to_fabsim",
        "format": FABSIM_TRACKS_FORMAT,
        "version": FABSIM_TRACKS_VERSION,
        "source": run_label(plan.run_dir),
        "role": plan.role,
        "model": "CoWTracker (facebookresearch/cowtracker)",
        "repo": COWTRACKER_REPO,
        "repo_commit": cowtracker_repo_commit(args),
        **checkpoint_identity(args),
        "adapter_version": ADAPTER_VERSION,
        "source_dimensions": [plan.source_w, plan.source_h],
        "inference_dimensions": [plan.inference_w, plan.inference_h],
        "reference_frame": plan.reference_frame_id,
        "reference_frame_sequence_index": plan.reference_seq_index,
        "sequence_rotated": plan.rotation_active,
        "confidence_mode": plan.confidence_mode,
        "coordinate_convention": plan.convention,
        "reference_error_limit_px": plan.reference_error_limit,
        "reference_errors_px": {
            p.id: float(reference_errors[i]) for i, p in enumerate(plan.points)
        },
        "frame_mapping": [
            {
                "sequence_index": f.sequence_index,
                "frame_id": f.frame_id,
                "rotated_position": int(
                    compute_rotation(plan.reference_seq_index, plan.frame_count)[i]
                ),
            }
            for i, f in enumerate(plan.frames)
        ],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    # The adapter is usually copied to the GPU host without its git history, so
    # the front end passes the commit it was uploaded from via the environment.
    adapter_commit = os.environ.get(ENV_ADAPTER_COMMIT) or discover_git_commit(
        os.path.dirname(os.path.abspath(__file__))
    )
    if adapter_commit:
        prov["adapter_git_commit"] = adapter_commit
    if runtime:
        prov["runtime"] = runtime

    return prov


def write_tracks_json(
    path: str,
    plan: RunPlan,
    track_src: np.ndarray,
    confidence: np.ndarray,
    provenance: dict[str, Any],
) -> None:
    frames_out: list[dict[str, Any]] = []
    for t, frame in enumerate(plan.frames):
        points_out = []
        for n, p in enumerate(plan.points):
            points_out.append(
                {
                    "id": p.id,
                    "x": float(track_src[t, n, 0]),
                    "y": float(track_src[t, n, 1]),
                    "confidence": _clean_probability(confidence[t, n]),
                }
            )
        frames_out.append({"frame_id": frame.frame_id, "points": points_out})
    doc = {
        "format": FABSIM_TRACKS_FORMAT,
        "version": FABSIM_TRACKS_VERSION,
        "provenance": provenance,
        "frames": frames_out,
    }
    _atomic_write_json(path, doc)


def _atomic_write_json(path: str, doc: Any) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError as exc:
        raise AdapterError(f"cannot write {path}: {exc}") from exc


def _clean_probability(value: Any) -> float:
    return round(float(value), 6)


def write_tracks_csv(
    path: str,
    plan: RunPlan,
    track_src: np.ndarray,
    confidence: np.ndarray,
    visibility: np.ndarray,
    raw_confidence: np.ndarray,
) -> None:
    lines = [
        "frame_id,track_id,x,y,confidence,visibility,raw_confidence,sequence_index"
    ]
    for t, frame in enumerate(plan.frames):
        for n, p in enumerate(plan.points):
            lines.append(
                f"{frame.frame_id},{p.id},{track_src[t, n, 0]:.9g},{track_src[t, n, 1]:.9g},"
                f"{_clean_probability(confidence[t, n]):.6f},"
                f"{_clean_probability(visibility[t, n]):.6f},"
                f"{_clean_probability(raw_confidence[t, n]):.6f},"
                f"{frame.sequence_index}"
            )
    try:
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as exc:
        raise AdapterError(f"cannot write {path}: {exc}") from exc


def write_sampled_npz(
    path: str,
    plan: RunPlan,
    track_src: np.ndarray,
    visibility: np.ndarray,
    raw_confidence: np.ndarray,
    confidence: np.ndarray,
    provenance: dict[str, Any],
) -> None:
    rot = compute_rotation(plan.reference_seq_index, plan.frame_count)
    arrays: dict[str, Any] = {
        "point_id": np.asarray([p.id for p in plan.points], dtype=object),
        "point_x": np.asarray([p.x for p in plan.points], dtype=np.float64),
        "point_y": np.asarray([p.y for p in plan.points], dtype=np.float64),
        "track_x": track_src[:, :, 0],
        "track_y": track_src[:, :, 1],
        "visibility": visibility,
        "raw_confidence": raw_confidence,
        "confidence": confidence,
        "frame_id": np.asarray([f.frame_id for f in plan.frames], dtype=np.int64),
        "sequence_index": np.asarray(
            [f.sequence_index for f in plan.frames], dtype=np.int64
        ),
        "rotated_position": rot,
        "metadata": np.asarray(json.dumps(provenance, indent=2)),
    }
    try:
        np.savez(path, **arrays)
    except OSError as exc:
        raise AdapterError(f"cannot write {path}: {exc}") from exc


def load_sampled_npz(path: str) -> dict[str, Any]:
    try:
        data = np.load(path, allow_pickle=True)
    except OSError as exc:
        raise AdapterError(f"cannot read sampled NPZ {path}: {exc}") from exc
    try:
        metadata = json.loads(str(data["metadata"]))
        points = [
            TrackPoint(id=str(pid), x=float(x), y=float(y))
            for pid, x, y in zip(
                data["point_id"], data["point_x"], data["point_y"], strict=False
            )
        ]
        out = {
            "points": points,
            "track_src": np.stack([data["track_x"], data["track_y"]], axis=-1),
            "visibility": np.asarray(data["visibility"]),
            "raw_confidence": np.asarray(data["raw_confidence"]),
            "confidence": np.asarray(data["confidence"]),
            "frame_id": np.asarray(data["frame_id"]),
            "sequence_index": np.asarray(data["sequence_index"]),
            "rotated_position": np.asarray(data["rotated_position"]),
            "metadata": metadata,
        }
    except (KeyError, ValueError, TypeError) as exc:
        raise AdapterError(
            f"{path}: not a cowtracker_to_fabsim sampled NPZ: {exc}"
        ) from exc

    return out


def write_dense_npz(
    path: str, dense: dict[str, np.ndarray], provenance: dict[str, Any]
) -> None:
    try:
        np.savez(
            path,
            track=dense["track"],
            vis=dense["vis"],
            conf=dense["conf"],
            metadata=np.asarray(json.dumps(provenance, indent=2)),
        )
    except OSError as exc:
        raise AdapterError(f"cannot write {path}: {exc}") from exc


def diagnostic_frame_indices(frame_count: int) -> list[int]:
    if frame_count == 0:
        return []
    indices = sorted(
        {min(frame_count - 1, (p * frame_count) // 100) for p in DIAGNOSTIC_PERCENTS}
    )
    # The 100% bucket resolves to the last frame index; keep the explicit
    # "first" too even when frame_count == 1.
    if 0 not in indices:
        indices.insert(0, 0)

    return indices


def annotate_frame(image: np.ndarray, points: list[dict[str, Any]]) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for pt in points:
        x, y = float(pt["x"]), float(pt["y"])
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 40, 40), width=2)
        draw.text((x + 9, y - 10), pt["id"], fill=(255, 255, 0))
        draw.text(
            (x + 9, y + 4),
            f"{pt['confidence']:.3f}",
            fill=(255, 200, 0),
        )

    return np.asarray(img)


def write_diagnostics_stills(
    plan: RunPlan,
    track_src: np.ndarray,
    confidence: np.ndarray,
    diagnostics_dir: str,
) -> list[str]:
    try:
        os.makedirs(diagnostics_dir, exist_ok=True)
    except OSError as exc:
        raise AdapterError(
            f"cannot create diagnostics dir {diagnostics_dir}: {exc}"
        ) from exc
    written: list[str] = []
    for t in diagnostic_frame_indices(plan.frame_count):
        frame = plan.frames[t]
        image = decode_image(frame.image_path)
        points = [
            {
                "id": p.id,
                "x": float(track_src[t, n, 0]),
                "y": float(track_src[t, n, 1]),
                "confidence": float(confidence[t, n]),
            }
            for n, p in enumerate(plan.points)
        ]
        annotated = annotate_frame(image, points)
        name = f"frame_{frame.frame_id:05d}_seq{t:03d}.png"
        out_path = os.path.join(diagnostics_dir, name)
        try:
            Image.fromarray(annotated).save(out_path)
        except OSError as exc:
            raise AdapterError(f"cannot write still {out_path}: {exc}") from exc
        written.append(out_path)

    return written


def write_diagnostic_video(
    plan: RunPlan, track_src: np.ndarray, confidence: np.ndarray, path: str
) -> None:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise AdapterError(
            "--diagnostic-video requires opencv-python (cv2). Install it "
            f"(pip install opencv-python) or drop the flag: {exc}"
        ) from exc
    writer = None
    for t, frame in enumerate(plan.frames):
        image = decode_image(frame.image_path)
        points = [
            {
                "id": p.id,
                "x": float(track_src[t, n, 0]),
                "y": float(track_src[t, n, 1]),
                "confidence": float(confidence[t, n]),
            }
            for n, p in enumerate(plan.points)
        ]
        annotated = annotate_frame(image, points)
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = bgr.shape[:2]
            try:
                writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h)
                )
            except Exception as exc:
                raise AdapterError(f"cannot open video writer {path}: {exc}") from exc
        writer.write(bgr)
    if writer is not None:
        writer.release()


def _resolve_inference_settings(
    args: argparse.Namespace, sampled_data: dict[str, Any] | None
) -> tuple[int, int, str]:
    stored = (sampled_data or {}).get("metadata") or {}
    stored_inf = stored.get("inference_dimensions")
    stored_conv = stored.get("coordinate_convention")

    if args.inference_width is not None:
        if stored_inf is not None and args.inference_width != stored_inf[0]:
            raise AdapterError(
                f"--inference-width {args.inference_width} conflicts with the "
                f"sampled NPZ's stored inference width {stored_inf[0]}"
            )
        inf_w = args.inference_width
    elif stored_inf is not None:
        inf_w = int(stored_inf[0])
    else:
        inf_w = DEFAULT_INFERENCE_WIDTH

    if args.inference_height is not None:
        if stored_inf is not None and args.inference_height != stored_inf[1]:
            raise AdapterError(
                f"--inference-height {args.inference_height} conflicts with the "
                f"sampled NPZ's stored inference height {stored_inf[1]}"
            )
        inf_h = args.inference_height
    elif stored_inf is not None:
        inf_h = int(stored_inf[1])
    else:
        inf_h = DEFAULT_INFERENCE_HEIGHT

    if args.coord_convention is not None:
        if stored_conv is not None and args.coord_convention != stored_conv:
            raise AdapterError(
                f"--coord-convention {args.coord_convention} conflicts with "
                f"the sampled NPZ's stored convention {stored_conv!r}"
            )
        convention = args.coord_convention
    elif stored_conv is not None:
        convention = stored_conv
    else:
        convention = PIXEL_CENTER

    return inf_w, inf_h, convention


def _validate_sampled_npz_against_run(
    data: dict[str, Any],
    frames: list[SequenceFrame],
    source_w: int,
    source_h: int,
    inf_w: int,
    inf_h: int,
    convention: str,
    role: str,
) -> None:
    meta = data.get("metadata") or {}
    problems: list[str] = []

    npz_seq = [int(v) for v in data["sequence_index"]]
    run_seq = [f.sequence_index for f in frames]
    if npz_seq != run_seq:
        problems.append(f"sequence_index: NPZ {npz_seq} != run {run_seq}")
    npz_fid = [int(v) for v in data["frame_id"]]
    run_fid = [f.frame_id for f in frames]
    if npz_fid != run_fid:
        problems.append(f"frame_id: NPZ {npz_fid} != run {run_fid}")
    if data["track_src"].shape[0] != len(frames):
        problems.append(
            f"track frame count {data['track_src'].shape[0]} != run frame count {len(frames)}"
        )
    stored_role = meta.get("role")
    if stored_role is not None and stored_role != role:
        problems.append(f"role: NPZ {stored_role!r} != CLI {role!r}")
    stored_src = meta.get("source_dimensions")
    if stored_src is not None and list(stored_src) != [source_w, source_h]:
        problems.append(
            f"source dimensions: NPZ {list(stored_src)} != run {source_w}x{source_h}"
        )
    stored_inf = meta.get("inference_dimensions")
    if stored_inf is not None and list(stored_inf) != [inf_w, inf_h]:
        problems.append(
            f"inference dimensions: NPZ {list(stored_inf)} != resolved {inf_w}x{inf_h}"
        )
    stored_conv = meta.get("coordinate_convention")
    if stored_conv is not None and stored_conv != convention:
        problems.append(
            f"coordinate convention: NPZ {stored_conv!r} != resolved {convention!r}"
        )
    if problems:
        raise AdapterError(
            "--sampled-input does not match the run/CLI; refusing to re-export "
            "with a mismatched frame/coordinate mapping:\n  " + "\n  ".join(problems)
        )


def _validate_points_match_npz(
    points: list[TrackPoint], data: dict[str, Any], atol: float = 1e-6
) -> None:
    npz_points = data["points"]
    npz_ids = [p.id for p in npz_points]
    ids = [p.id for p in points]
    if ids != npz_ids:
        raise AdapterError(
            "--sampled-input was produced for point ids in order "
            f"{npz_ids}, but --points/grid supplies {ids}; refusing to "
            "relabel sampled coordinates under different identities"
        )
    npz_xy = np.asarray([[p.x, p.y] for p in npz_points], dtype=np.float64)
    xy = np.asarray([[p.x, p.y] for p in points], dtype=np.float64)
    dist = np.hypot(xy[:, 0] - npz_xy[:, 0], xy[:, 1] - npz_xy[:, 1])
    bad = [i for i, d in enumerate(dist) if d > atol]
    if bad:
        detail = "\n".join(
            f"  point {points[i].id!r}: supplied ({points[i].x:.6f}, "
            f"{points[i].y:.6f}) vs sampled ({npz_points[i].x:.6f}, "
            f"{npz_points[i].y:.6f})"
            for i in bad
        )
        raise AdapterError(
            f"{len(bad)} supplied point coordinate(s) differ from the sampled "
            f"NPZ by more than {atol} px:\n{detail}"
        )


def _resolve_points(
    args: argparse.Namespace,
    source_w: int,
    source_h: int,
    sampled_data: dict[str, Any] | None,
) -> tuple[list[TrackPoint], str]:
    if args.points or args.grid_cols or args.grid_rows or args.grid_spacing:
        if args.points:
            points = load_points(args.points)
            points_mode = "points-file"
        else:
            if args.grid_spacing:
                cols = args.grid_cols or 0
                rows = args.grid_rows or 0
            else:
                if not (args.grid_cols and args.grid_rows):
                    raise AdapterError(
                        "grid mode requires both --grid-cols and --grid-rows "
                        "(or --grid-spacing SX SY); a single dimension is "
                        "ambiguous and a square grid is never assumed"
                    )
                cols, rows = args.grid_cols, args.grid_rows
            roi = tuple(args.grid_roi) if args.grid_roi else None
            spacing = tuple(args.grid_spacing) if args.grid_spacing else None
            points = grid_points(cols, rows, source_w, source_h, roi, spacing)
            points_mode = "grid"
        if sampled_data is not None:
            _validate_points_match_npz(points, sampled_data)

        return points, points_mode
    if sampled_data is not None:
        return list(sampled_data["points"]), "sampled-input"
    raise AdapterError(
        "no query points: pass --points <file> (primary mode), "
        "--grid-cols/--grid-rows (grid mode), or --sampled-input <npz> "
        "for a self-contained re-export"
    )


def _resolve_reference_frame(
    args: argparse.Namespace,
    frames: list[SequenceFrame],
    sampled_data: dict[str, Any] | None,
) -> tuple[int, int]:
    if args.reference_frame is None:
        if sampled_data is not None:
            stored_ref = (sampled_data.get("metadata") or {}).get(
                "reference_frame_sequence_index"
            )
            matches = [f for f in frames if f.sequence_index == stored_ref]
            if not matches:
                raise AdapterError(
                    f"sampled NPZ reference frame (sequence index {stored_ref}) "
                    "is not among the included frames"
                )

            return matches[0].frame_id, matches[0].sequence_index

        return frames[0].frame_id, frames[0].sequence_index
    matches = [f for f in frames if f.frame_id == args.reference_frame]
    if not matches:
        raise AdapterError(
            f"--reference-frame {args.reference_frame} is not among the "
            f"included frames (ids: {[f.frame_id for f in frames]})"
        )
    reference_frame_id = matches[0].frame_id
    reference_seq_index = matches[0].sequence_index
    if sampled_data is not None:
        stored_ref = (sampled_data.get("metadata") or {}).get("reference_frame")
        if stored_ref is not None and reference_frame_id != stored_ref:
            raise AdapterError(
                f"--reference-frame {reference_frame_id} conflicts with the "
                f"sampled NPZ reference frame {stored_ref}"
            )

    return reference_frame_id, reference_seq_index


def preflight(args: argparse.Namespace) -> RunPlan:
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise AdapterError(f"run directory not found: {run_dir}")

    frames, skipped = build_sequence(run_dir, args.role, args.skip_missing)
    source_w, source_h = probe_sequence_dimensions(frames)

    sampled_data = None
    if args.sampled_input:
        sampled_data = load_sampled_npz(args.sampled_input)

    inf_w, inf_h, convention = _resolve_inference_settings(args, sampled_data)
    if sampled_data is not None:
        _validate_sampled_npz_against_run(
            sampled_data,
            frames,
            source_w,
            source_h,
            inf_w,
            inf_h,
            convention,
            args.role,
        )

    points, points_mode = _resolve_points(args, source_w, source_h, sampled_data)
    reference_frame_id, reference_seq_index = _resolve_reference_frame(
        args, frames, sampled_data
    )
    if sampled_data is not None:
        expected_rot = [
            int(v) for v in compute_rotation(reference_seq_index, len(frames))
        ]
        npz_rot = [int(v) for v in sampled_data["rotated_position"]]
        if expected_rot != npz_rot:
            raise AdapterError(
                f"sampled NPZ rotated_position {npz_rot} does not match the "
                f"rotation {expected_rot} implied by the resolved reference "
                f"frame (frame_id {reference_frame_id}, sequence index "
                f"{reference_seq_index})"
            )

    rotation_active = reference_seq_index != frames[0].sequence_index
    scale_x = inf_w / source_w
    scale_y = inf_h / source_h

    output_path = os.path.abspath(args.output or os.path.join(run_dir, "tracks.json"))
    csv_output = os.path.abspath(args.csv_output) if args.csv_output else None
    save_sampled = os.path.abspath(args.save_sampled) if args.save_sampled else None
    save_dense = os.path.abspath(args.save_dense) if args.save_dense else None
    diagnostics_dir = (
        os.path.abspath(args.diagnostics_dir) if args.diagnostics_dir else None
    )
    diagnostic_video = (
        os.path.abspath(args.diagnostic_video) if args.diagnostic_video else None
    )

    plan = RunPlan(
        run_dir=run_dir,
        role=args.role,
        frames=frames,
        skipped=skipped,
        source_w=source_w,
        source_h=source_h,
        inference_w=inf_w,
        inference_h=inf_h,
        scale_x=scale_x,
        scale_y=scale_y,
        points=points,
        points_mode=points_mode,
        reference_frame_id=reference_frame_id,
        reference_seq_index=reference_seq_index,
        rotation_active=rotation_active,
        convention=convention,
        confidence_mode=args.confidence_mode,
        reference_error_limit=args.reference_error_limit,
        output_path=output_path,
        csv_output=csv_output,
        save_sampled=save_sampled,
        save_dense=save_dense,
        diagnostics_dir=diagnostics_dir,
        diagnostic_video=diagnostic_video,
        device=args.device,
    )
    plan.notes.extend(
        validate_points_in_image(points, source_w, source_h, inf_w, inf_h, convention)
    )
    if rotation_active:
        plan.notes.append(
            f"reference frame {reference_frame_id} is sequence index "
            f"{reference_seq_index}; the sequence will be rotated so it is the "
            "model input frame 0, and output frames remapped back "
            "(see provenance frame_mapping.rotated_position)"
        )

    return plan


def print_dry_run_report(plan: RunPlan, args: argparse.Namespace) -> None:
    sep = "=" * 68
    print(sep)
    print("CoWTracker -> fabsim adapter: dry run (no inference, no torch)")
    print(sep)
    print(f"run directory          : {plan.run_dir}")
    print(f"role                   : {plan.role}")
    print(
        f"frames (frames.json)   : {plan.frame_count} included, {len(plan.skipped)} skipped"
    )
    for s in plan.skipped:
        print(
            f"  skipped seq {s['sequence_index']} frame_id {s['frame_id']}: {s['reason']}"
        )
    print(f"source dimensions      : {plan.source_w} x {plan.source_h} (w x h)")
    print(f"inference dimensions   : {plan.inference_w} x {plan.inference_h} (w x h)")
    print(
        f"scales                 : x {plan.scale_x:.6f}, y {plan.scale_y:.6f} ({plan.convention} convention)"
    )
    if plan.points:
        x0, y0 = plan.points[0].x, plan.points[0].y
        xi0, yi0 = source_to_inference(
            x0,
            y0,
            plan.source_w,
            plan.source_h,
            plan.inference_w,
            plan.inference_h,
            plan.convention,
        )
        print(f"points                 : {len(plan.points)} ({plan.points_mode})")
        print(
            f"  example transform    : ({x0:.3f}, {y0:.3f}) -> inference ({xi0:.3f}, {yi0:.3f})"
        )
    print(
        f"reference frame        : {plan.reference_frame_id} (sequence index {plan.reference_seq_index})"
    )
    print(f"sequence rotation      : {'yes' if plan.rotation_active else 'no'}")
    print("frame mapping (sequence_index -> frame_id -> rotated_position):")
    rot = compute_rotation(plan.reference_seq_index, plan.frame_count)
    for i, f in enumerate(plan.frames):
        print(
            f"  seq {f.sequence_index:>3} -> frame_id {f.frame_id:>3} -> rot {int(rot[i]):>3}"
        )
    print(f"output (fabsim-tracks) : {plan.output_path}")
    for label, p in (
        ("csv output", plan.csv_output),
        ("sampled npz", plan.save_sampled),
        ("dense npz", plan.save_dense),
        ("diagnostics dir", plan.diagnostics_dir),
        ("diagnostic video", plan.diagnostic_video),
    ):
        if p:
            print(f"{label:<22} : {p}")
    print(f"reference error limit  : {plan.reference_error_limit} px")
    print(f"confidence mode        : {plan.confidence_mode}")
    print(f"inference device       : {plan.device} (CUDA required unless --device cpu)")
    print("torch / cowtracker     : not imported (dry run)")
    for note in plan.notes:
        print(f"note                   : {note}")


def _import_torch_and_cowtracker(root: str | None) -> tuple[Any, Any, str]:
    if root:
        root = os.path.abspath(root)
        if not os.path.isdir(os.path.join(root, "cowtracker")) and not os.path.isfile(
            os.path.join(root, "cowtracker.py")
        ):
            raise AdapterError(
                f"--cowtracker-root {root} does not look like a CoWTracker "
                f"checkout (no cowtracker/ package inside)"
            )
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        import torch  # noqa: F401

        torch_version = torch.__version__
    except ImportError as exc:
        raise AdapterError(
            "PyTorch is required for inference but could not be imported. "
            "Set up a CUDA environment with server/bootstrap.sh (see "
            "docs/self_hosting_tracker.md). Non-inference paths (--help, --dry-run, "
            "tests) never need torch."
        ) from exc
    try:
        import cowtracker  # noqa: F401
    except ImportError as exc:
        raise AdapterError(
            "the CoWTracker package could not be imported. Either pip install "
            "cowtracker, or point --cowtracker-root at a checkout of "
            f"{COWTRACKER_REPO} (pinned revision {COWTRACKER_COMMIT})."
        ) from exc

    return torch, cowtracker, torch_version


def cowtracker_repo_commit(args: argparse.Namespace) -> str:
    if args.cowtracker_root:
        commit = discover_git_commit(args.cowtracker_root)
        if commit:
            return commit

    return COWTRACKER_COMMIT


def discover_git_commit(directory: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", directory, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commit = out.stdout.strip()

    return commit or None


def build_video_tensor(plan: RunPlan, torch_mod: Any, device: str) -> Any:
    frames_float: list[np.ndarray] = []
    for frame in plan.frames:
        image = decode_image(frame.image_path)  # [H, W, 3] float32
        resized = pixel_center_resize(image, plan.inference_h, plan.inference_w)
        frames_float.append(resized)
    video = np.stack(frames_float, axis=0)  # [T, Hi, Wi, 3]
    tensor = torch_mod.from_numpy(video).permute(0, 3, 1, 2).float().to(device)

    return tensor


def run_inference(
    plan: RunPlan, args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    torch_mod, cowtracker_mod, torch_version = _import_torch_and_cowtracker(
        args.cowtracker_root
    )

    if args.device == "cuda":
        if not torch_mod.cuda.is_available():
            raise AdapterError(
                "inference device is CUDA (default) but torch reports no CUDA "
                "available. Run on a CUDA machine, or explicitly request "
                "--device cpu (not recommended for this model)."
            )
        device = "cuda"
    else:
        device = "cpu"

    dtype_str = args.dtype
    if dtype_str is None:
        dtype_str = "float16" if device == "cuda" else "float32"
    dtype = {
        "float16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "float32": torch_mod.float32,
    }[dtype_str]

    if args.windowed:
        model_cls_name = "CoWTrackerWindowed"
    else:
        model_cls_name = "CoWTracker"
        if plan.frame_count > 200:
            raise AdapterError(
                f"this sequence has {plan.frame_count} frames; the non-windowed "
                "CoWTracker model is designed for short videos. Use --windowed "
                "for long sequences."
            )

    print(f"torch version          : {torch_version}")
    print(f"CUDA available         : {torch_mod.cuda.is_available()}")
    print(f"inference device       : {device}")
    print(f"inference dtype        : {dtype_str}")
    default_checkpoint = (
        f"<default: {COWTRACKER_MODEL_REPO}/{COWTRACKER_MODEL_FILENAME}>"
    )
    print(f"checkpoint             : {args.checkpoint or default_checkpoint}")
    print(f"model class            : {model_cls_name} (repo {COWTRACKER_REPO})")

    model_cls = getattr(cowtracker_mod, model_cls_name)
    model = model_cls.from_checkpoint(args.checkpoint, device=device, dtype=dtype)

    # Explicit sequence rotation: the requested reference frame becomes input
    # frame 0 so the dense reference grid is anchored exactly there.
    rot = compute_rotation(plan.reference_seq_index, plan.frame_count)
    order = list(np.argsort(rot))  # original sequence index per rotated position
    video = build_video_tensor(plan, torch_mod, device)
    video_rotated = video[order]

    with torch_mod.no_grad():
        autocast_ctx = (
            torch_mod.amp.autocast(device_type="cuda", dtype=dtype)
            if device == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            # Upstream API: `queries` is unused by the model (dense tracker).
            predictions = model.forward(video=video_rotated, queries=None)
            track = (
                predictions["track"][0].detach().float().cpu().numpy()
            )  # [T, Hi, Wi, 2]
            vis = predictions["vis"][0].detach().float().cpu().numpy()  # [T, Hi, Wi]
            conf = predictions["conf"][0].detach().float().cpu().numpy()  # [T, Hi, Wi]

    if track.shape[0] != plan.frame_count:
        raise AdapterError(
            f"model output frame count {track.shape[0]} != input frame count {plan.frame_count}"
        )
    if track.shape[1] != plan.inference_h or track.shape[2] != plan.inference_w:
        raise AdapterError(
            f"model output spatial size {track.shape[2]}x{track.shape[1]} != "
            f"inference size {plan.inference_w}x{plan.inference_h}"
        )

    # Remap from rotated order back to the original sequence order.
    dense = {
        "track": np.asarray([track[int(p)] for p in rot], dtype=np.float32),
        "vis": np.asarray([vis[int(p)] for p in rot], dtype=np.float32),
        "conf": np.asarray([conf[int(p)] for p in rot], dtype=np.float32),
    }

    return dense, collect_runtime_info(torch_mod, device)


def sample_and_export(
    plan: RunPlan,
    args: argparse.Namespace,
    dense: dict[str, np.ndarray],
    runtime: dict[str, Any] | None = None,
) -> None:
    T = plan.frame_count
    N = len(plan.points)

    query_src = np.asarray([[p.x, p.y] for p in plan.points], dtype=np.float64)
    query_inf = np.asarray(
        [
            source_to_inference(
                p.x,
                p.y,
                plan.source_w,
                plan.source_h,
                plan.inference_w,
                plan.inference_h,
                plan.convention,
            )
            for p in plan.points
        ],
        dtype=np.float64,
    )

    # Bilinear sampling of the dense track / vis / conf at each query.
    track_inf = sample_dense_field(
        dense["track"], query_inf[:, 0], query_inf[:, 1]
    )  # [T, N, 2]
    visibility = sample_dense_field(
        dense["vis"], query_inf[:, 0], query_inf[:, 1]
    )  # [T, N]
    raw_confidence = sample_dense_field(
        dense["conf"], query_inf[:, 0], query_inf[:, 1]
    )  # [T, N]

    # Convert every tracked coordinate back to original source pixels (kept
    # as float64 so full positional precision is preserved in the outputs).
    track_src = np.empty((T, N, 2), dtype=np.float64)
    for t in range(T):
        for n in range(N):
            xs, ys = inference_to_source(
                float(track_inf[t, n, 0]),
                float(track_inf[t, n, 1]),
                plan.source_w,
                plan.source_h,
                plan.inference_w,
                plan.inference_h,
                plan.convention,
            )
            track_src[t, n, 0] = xs
            track_src[t, n, 1] = ys

    confidence = combine_confidence(visibility, raw_confidence, plan.confidence_mode)

    # Reference reconstruction error: the reference frame's sampled track must
    # reconstruct each query position.  Every offending point is reported in
    # one go instead of aborting on the first one.
    ref_errors = reference_error_px(track_src, query_src, plan.reference_seq_index)
    max_err = float(np.max(ref_errors)) if N else 0.0
    print(
        f"reference reconstruction error: max {max_err:.6f} px (limit {plan.reference_error_limit} px)"
    )
    over = [
        n
        for n, p in enumerate(plan.points)
        if ref_errors[n] > plan.reference_error_limit
    ]
    if over:
        detail = "\n".join(
            f"  point {plan.points[n].id!r}: {ref_errors[n]:.4f} px" for n in over
        )
        raise AdapterError(
            f"{len(over)} point(s) exceed --reference-error-limit "
            f"{plan.reference_error_limit} px (max {max_err:.4f} px):\n{detail}"
        )

    provenance = build_provenance(plan, args, ref_errors, runtime)

    write_tracks_json(plan.output_path, plan, track_src, confidence, provenance)
    print(f"wrote {plan.output_path}")
    if plan.csv_output:
        write_tracks_csv(
            plan.csv_output, plan, track_src, confidence, visibility, raw_confidence
        )
        print(f"wrote {plan.csv_output}")
    if plan.save_sampled:
        write_sampled_npz(
            plan.save_sampled,
            plan,
            track_src,
            visibility,
            raw_confidence,
            confidence,
            provenance,
        )
        print(f"wrote {plan.save_sampled}")
    if plan.save_dense:
        write_dense_npz(plan.save_dense, dense, provenance)
        print(f"wrote {plan.save_dense}")
    if plan.diagnostics_dir:
        written = write_diagnostics_stills(
            plan, track_src, confidence, plan.diagnostics_dir
        )
        print(f"wrote {len(written)} diagnostic still(s) to {plan.diagnostics_dir}")
    if plan.diagnostic_video:
        write_diagnostic_video(plan, track_src, confidence, plan.diagnostic_video)
        print(f"wrote {plan.diagnostic_video}")

    print(
        f"emitted {T} frame(s) x {N} point(s) with confidence mode {plan.confidence_mode}"
    )


def export_from_sampled(
    plan: RunPlan, args: argparse.Namespace, data: dict[str, Any]
) -> None:
    if plan.save_dense:
        raise AdapterError(
            "--save-dense requires a fresh inference run; it cannot be "
            "reconstructed from a --sampled-input NPZ (only the sampled "
            "coordinates are stored there)"
        )
    track_src = data["track_src"]
    visibility = data["visibility"]
    raw_confidence = data["raw_confidence"]
    confidence = data["confidence"]
    provenance = data["metadata"]

    if track_src.shape[0] != plan.frame_count:
        raise AdapterError(
            f"sampled NPZ frame count {track_src.shape[0]} != included frame "
            f"count {plan.frame_count} in {plan.run_dir}"
        )

    write_tracks_json(plan.output_path, plan, track_src, confidence, provenance)
    print(f"wrote {plan.output_path} (from {args.sampled_input})")
    if plan.csv_output:
        write_tracks_csv(
            plan.csv_output, plan, track_src, confidence, visibility, raw_confidence
        )
        print(f"wrote {plan.csv_output}")
    if plan.save_sampled:
        write_sampled_npz(
            plan.save_sampled,
            plan,
            track_src,
            visibility,
            raw_confidence,
            confidence,
            provenance,
        )
        print(f"wrote {plan.save_sampled}")
    if plan.diagnostics_dir:
        written = write_diagnostics_stills(
            plan, track_src, confidence, plan.diagnostics_dir
        )
        print(f"wrote {len(written)} diagnostic still(s) to {plan.diagnostics_dir}")
    if plan.diagnostic_video:
        write_diagnostic_video(plan, track_src, confidence, plan.diagnostic_video)
        print(f"wrote {plan.diagnostic_video}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cowtracker_to_fabsim.py",
        description=(
            "Official CoWTracker -> fabsim tracking adapter. Converts a fabsim "
            "run (frames.json + source images) into the fabsim-tracks JSON "
            "schema. --help and --dry-run never import torch/cowtracker."
        ),
    )
    parser.add_argument("run_dir", help="fabsim run directory containing frames.json")
    parser.add_argument(
        "--role", default="diffuse", help="image_<role> key to track (default: diffuse)"
    )
    parser.add_argument(
        "--points", help="explicit point file {points:[{id,x,y}]} (primary mode)"
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=0,
        help="grid mode: number of columns (requires --grid-rows too)",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=0,
        help="grid mode: number of rows (requires --grid-cols too)",
    )
    parser.add_argument(
        "--grid-roi",
        nargs=4,
        type=float,
        metavar=("X", "Y", "W", "H"),
        help="grid ROI in source pixels (default: whole image)",
    )
    parser.add_argument(
        "--grid-spacing",
        nargs=2,
        type=float,
        metavar=("SX", "SY"),
        help="grid step in source pixels (overrides --grid-cols/--grid-rows)",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        help="fabsim frame_id of the tracking reference frame (default: first included frame)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="skip frames lacking the requested role instead of failing",
    )
    parser.add_argument(
        "--cowtracker-root",
        help="path to a CoWTracker checkout (else the cowtracker package must be installed)",
    )
    parser.add_argument(
        "--checkpoint", help="model checkpoint path (default: HuggingFace download)"
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="inference device (default: cuda; fails clearly without it unless --device cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=None,
        help="inference dtype (default: float16 on CUDA, float32 on CPU)",
    )
    parser.add_argument(
        "--inference-width",
        type=int,
        default=None,
        help="inference width (default: 560, or the sampled NPZ's stored value with --sampled-input)",
    )
    parser.add_argument(
        "--inference-height",
        type=int,
        default=None,
        help="inference height (default: 336, or the sampled NPZ's stored value with --sampled-input)",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="use CoWTrackerWindowed for long sequences",
    )
    parser.add_argument(
        "--coord-convention",
        choices=CONVENTIONS,
        default=None,
        help="coordinate convention (default: pixel_center, or the sampled "
        "NPZ's stored convention with --sampled-input)",
    )
    parser.add_argument(
        "--confidence-mode",
        choices=CONFIDENCE_MODES,
        default="vis*conf",
        help="emitted confidence (default: vis*conf)",
    )
    parser.add_argument(
        "--reference-error-limit",
        type=float,
        default=DEFAULT_REFERENCE_ERROR_LIMIT,
        help="per-point reference reconstruction error limit in source px",
    )
    parser.add_argument(
        "--output",
        help="fabsim-tracks JSON output path (default: <run_dir>/tracks.json for auto-discovery)",
    )
    parser.add_argument(
        "--csv-output",
        help="also write a long-table CSV with visibility/raw_confidence/sequence_index",
    )
    parser.add_argument(
        "--save-sampled",
        help="write the sampled intermediate NPZ (points, coords, raw vis/conf, mapping)",
    )
    parser.add_argument(
        "--sampled-input",
        help="re-run conversion/export from a --save-sampled NPZ without "
        "inference; self-contained (points, frame/reference mapping and "
        "dimensions come from the NPZ), any supplied --points/grid must match",
    )
    parser.add_argument(
        "--save-dense",
        help="write the dense model output NPZ (inference space) -- explicit opt-in",
    )
    parser.add_argument(
        "--diagnostics-dir",
        help="write diagnostic stills (first/25/50/75/last) over original images",
    )
    parser.add_argument(
        "--diagnostic-video",
        help="write an annotated video over original images (requires cv2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the full traceback of unexpected failures",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="full preflight report without importing torch/cowtracker",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        plan = preflight(args)
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)

        return 1

    if args.dry_run:
        print_dry_run_report(plan, args)

        return 0

    try:
        if args.sampled_input:
            data = load_sampled_npz(args.sampled_input)
            export_from_sampled(plan, args, data)

            return 0
        dense, runtime = run_inference(plan, args)
        sample_and_export(plan, args, dense, runtime)

        return 0
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)

        return 1
    except Exception as exc:  # broad: surface the real failure with context
        print(f"error: unexpected failure: {exc!r}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        else:
            print("(re-run with --verbose for the traceback)", file=sys.stderr)

        return 1


if __name__ == "__main__":
    sys.exit(main())
