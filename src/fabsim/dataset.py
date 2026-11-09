from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: Frame keys of the three photometric-stereo captures, in LED order.
PS_KEYS = ("image_ps_1", "image_ps_2", "image_ps_3")
#: The two motion branches of an orientation.
BRANCHES = ("loading", "release")

_RUN_NAME = re.compile(r"^(?P<orientation>.+)_(?P<branch>loading|release)$")
#: A run folder with a numeric suffix, as the rig names a run when the plain name is already taken.
_SUFFIXED_RUN_NAME = re.compile(r"^(?P<name>.+_(?:loading|release))_\d+$")
#: Entries whose presence makes a folder look like a run folder.
_RUN_MARKERS = ("frames.json", "force.json", "session.json", "images")


@dataclass(frozen=True)
class Run:
    path: Path
    orientation: str
    branch: str

    @property
    def name(self) -> str:
        return self.path.name


def discover_runs(project: Path, orientations: list[str] | None = None) -> list[Run]:
    project = Path(project)
    runs = []
    for entry in sorted(project.iterdir()) if project.is_dir() else []:
        match = _RUN_NAME.match(entry.name)
        if not (match and entry.is_dir() and (entry / "frames.json").is_file()):
            continue

        if orientations and match["orientation"] not in orientations:
            continue

        runs.append(Run(entry, match["orientation"], match["branch"]))

    runs.sort(key=lambda r: (r.orientation, BRANCHES.index(r.branch)))

    return runs


def ignored_run_folders(project: Path) -> list[tuple[str, str]]:
    project = Path(project)
    ignored = []
    for entry in sorted(project.iterdir()) if project.is_dir() else []:
        if entry.name.startswith(".") or not entry.is_dir():
            continue

        if not any((entry / marker).exists() for marker in _RUN_MARKERS):
            continue

        if _RUN_NAME.match(entry.name):
            if not (entry / "frames.json").is_file():
                ignored.append((entry.name, "no frames.json"))
            continue

        suffixed = _SUFFIXED_RUN_NAME.match(entry.name)
        hint = f"; rename it to replace {suffixed['name']}" if suffixed else ""
        ignored.append(
            (entry.name, f"name is not <orientation>_<loading|release>{hint}")
        )

    return ignored


def warn_ignored_run_folders(project: Path, file=None) -> list[tuple[str, str]]:
    ignored = ignored_run_folders(project)
    for name, reason in ignored:
        print(f"warning: {name}: not used as a run ({reason})", file=file or sys.stderr)

    return ignored


def discover_orientations(
    project: Path, orientations: list[str] | None = None
) -> dict[str, dict[str, Run]]:
    grouped: dict[str, dict[str, Run]] = {}
    for run in discover_runs(project, orientations):
        grouped.setdefault(run.orientation, {})[run.branch] = run

    return grouped


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_session(run: Path) -> dict:
    path = Path(run) / "session.json"

    return _load_json(path) if path.is_file() else {}


def load_frames(run: Path) -> list[dict]:
    return list(_load_json(Path(run) / "frames.json")["frames"])


def load_force(run: Path) -> dict[int, dict]:
    path = Path(run) / "force.json"
    if not path.is_file():
        return {}

    readings = {}
    for entry in _load_json(path).get("forces", []):
        value = entry.get("force_n")
        if (
            entry.get("status") == "ok"
            and isinstance(value, (int, float))
            and math.isfinite(value)
        ):
            readings[int(entry["index"])] = entry

    return readings


def force_for_frame(frame: dict, forces: dict[int, dict]) -> dict | None:
    index = frame.get("force_json_index")

    return forces.get(int(index)) if index is not None else None


def image_path(run: Path, name: str) -> Path:
    return Path(run) / "images" / name


_PNG_END = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def image_problem(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return "file is missing"
    if path.suffix.lower() == ".png":
        with path.open("rb") as handle:
            handle.seek(0, 2)
            if handle.tell() < len(_PNG_END):
                return "file is truncated"
            handle.seek(-len(_PNG_END), 2)
            if handle.read() != _PNG_END:
                return "PNG is truncated or damaged (no end chunk)"

    return None


def has_ps(frame: dict) -> bool:
    return all(frame.get(key) for key in PS_KEYS)


def frame_id(frame: dict, position: int) -> int:
    return int(frame.get("frame_id", position))
