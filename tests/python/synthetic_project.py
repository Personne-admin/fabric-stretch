from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from fabsim import photometric as ph

SIZE = 320  # image side in px
COLS, ROWS, SPACING, ORIGIN = 8, 6, 24.0, 40.0
STEP = 0.04  # stretch per loading frame
RELEASE_RESIDUAL = 1.03
GAINS = np.array([1.25, 0.8, 0.95])
LOADING_ZERO, RELEASE_ZERO, PER_COUNT = -628, -279, 0.0125
LOADING_FORCE = [
    -4.3,
    -3.2,
    -1.9,
    -0.4,
]  # N as stored for the loading frames (frame 0 is the tare)
RELEASE_FORCE = [
    1.5,
    0.9,
    0.3,
    0.1,
]  # N the release frames read with the Loading zero_counts
CLAMP_WIDTH_M = 0.2


def stored_force(branch: str, i: int) -> float:
    if branch == "loading":
        return LOADING_FORCE[i]

    return RELEASE_FORCE[i] + (RELEASE_ZERO - LOADING_ZERO) * PER_COUNT


def texture(seed: int = 3) -> np.ndarray:
    noise = np.random.default_rng(seed).random((SIZE, SIZE * 2)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.0)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())

    return 0.25 + 0.55 * smooth


def render(tex: np.ndarray, stretch: float) -> np.ndarray:
    xs = ORIGIN + (np.arange(SIZE, dtype=np.float32) - ORIGIN) / stretch
    map_x = np.tile(xs, (SIZE, 1)).astype(np.float32)
    map_y = np.tile(np.arange(SIZE, dtype=np.float32)[:, None], (1, SIZE))

    return cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR)


def grid_points(stretch: float) -> list[dict]:
    return [
        {
            "id": f"grid_{r * COLS + c}",
            "x": ORIGIN + c * SPACING * stretch,
            "y": ORIGIN + r * SPACING,
            "confidence": 1.0,
        }
        for r in range(ROWS)
        for c in range(COLS)
    ]


def stretches(branch: str, count: int) -> list[float]:
    end = 1 + STEP * (count - 1)
    if branch == "loading":
        return [1 + STEP * i for i in range(count)]

    return list(np.linspace(end, RELEASE_RESIDUAL, count))


def write_run(
    path: Path,
    branch: str,
    count: int = 4,
    ps: bool = True,
    extra_frame_without_force: bool = False,
    brighten: float = 0.0,
    diffuse_gain: float = 1.0,
    ramp: float = 0.0,
    gain_drift: float = 0.0,
    invert_diffuse: bool = False,
):
    (path / "images").mkdir(parents=True)
    tex = texture()
    cfg = ph.load_light_config()
    factors = stretches(branch, count + int(extra_frame_without_force))
    factors_max = 1 + STEP * (count - 1)
    zero = LOADING_ZERO if branch == "loading" else RELEASE_ZERO
    frames, forces, tracked = [], [], []
    for i, stretch in enumerate(factors):
        gray = render(tex, stretch) * (1 + brighten * (stretch - 1) / (factors_max - 1))
        shown = np.clip(gray * diffuse_gain, 0, 1)
        shown = 1 - shown if invert_diffuse else shown
        diffuse = np.rint(np.dstack([shown] * 3) * 255).astype(np.uint8)
        frame = {
            "frame_id": i,
            "force_json_index": (
                2 * i + 1 if i < count else 99
            ),  # indices 1, 3, 5, ...: never the list position
            "displacement_mm": 1.2 * i,
            "image_diffuse": f"img_{i:04d}_diffuse.png",
        }
        cv2.imwrite(
            str(path / "images" / frame["image_diffuse"]),
            cv2.cvtColor(diffuse, cv2.COLOR_RGB2BGR),
        )
        if ps:
            lean = ramp * np.tile(np.linspace(-1, 1, SIZE, dtype=np.float32), (SIZE, 1))
            normals = np.dstack([lean, np.zeros((SIZE, SIZE)), np.ones((SIZE, SIZE))])
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            gains = GAINS * [1, 1 - gain_drift * (stretch - 1) / (factors_max - 1), 1]
            shots = ph.render(gray, normals, cfg, gains)
            for k, key in enumerate(("image_ps_1", "image_ps_2", "image_ps_3")):
                frame[key] = f"img_{i:04d}_ps_{k + 1}.png"
                cv2.imwrite(
                    str(path / "images" / frame[key]),
                    np.rint(shots[..., k] * 255).astype(np.uint8),
                )
        frames.append(frame)
        tracked.append({"frame_id": i, "points": grid_points(stretch)})
        if i < count:
            forces.append(
                {
                    "index": 2 * i + 1,
                    "status": "ok",
                    "force_n": stored_force(branch, i),
                    "raw_counts": zero - stored_force(branch, i) / PER_COUNT,
                    "zero_counts": zero,
                    "calibration_n_per_count": PER_COUNT,
                }
            )
            forces.append({"index": 2 * i + 2, "status": "timeout", "force_n": None})
    (path / "frames.json").write_text(
        json.dumps({"clamp_width_m": CLAMP_WIDTH_M, "frames": frames})
    )
    (path / "force.json").write_text(json.dumps({"forces": forces}))
    (path / "session.json").write_text(
        json.dumps({"sample_id": "synthetic", "branch": branch, "count": count})
    )
    (path / "tracks.json").write_text(
        json.dumps({"provenance": {"reference_frame": 0}, "frames": tracked})
    )


def make_project(root: Path, orientations=("warp",), **kwargs) -> Path:
    for orientation in orientations:
        write_run(
            root / f"{orientation}_loading",
            "loading",
            extra_frame_without_force=True,
            **kwargs,
        )
        write_run(root / f"{orientation}_release", "release", **kwargs)
    write_run(root / "weft_loading", "loading", **kwargs)

    return root
