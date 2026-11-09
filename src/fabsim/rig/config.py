from __future__ import annotations

import json
import logging
import logging.handlers
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

APP_DIR = "fabsim-rig"
PRESETS_FILE = "presets.json"
LED_CONFIG_FILE = "led_config.json"
COLOR_CALIBRATION_FILE = "color_calibration.json"
STATE_FILE = "state.json"
LOG_FILE = "rig.log"

#: Files that ``seed_defaults`` copies from the package.
DEFAULT_FILES = (PRESETS_FILE, LED_CONFIG_FILE, COLOR_CALIBRATION_FILE)


class ConfigError(RuntimeError):
    pass


def _home() -> Path:
    return Path.home()


def config_dir() -> Path:
    override = os.environ.get("FABSIM_RIG_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or _home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = _home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or _home() / ".config")

    return base / APP_DIR


def log_dir() -> Path:
    override = os.environ.get("FABSIM_RIG_LOG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or _home() / "AppData" / "Local")

        return base / APP_DIR / "Logs"
    if sys.platform == "darwin":
        return _home() / "Library" / "Logs" / APP_DIR

    return Path(os.environ.get("XDG_STATE_HOME") or _home() / ".local" / "state") / APP_DIR


def packaged_default(name: str) -> Path:
    return Path(str(resources.files("fabsim.rig") / "defaults" / name))


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def load_packaged_json(name: str):
    return _read_json(packaged_default(name))


def load_json(name: str, directory: Path | None = None):
    path = (directory or config_dir()) / name

    return _read_json(path if path.is_file() else packaged_default(name))


def save_json(name: str, data, directory: Path | None = None) -> Path:
    path = (directory or config_dir()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

    return path


def seed_defaults(directory: Path | None = None, overwrite: bool = False) -> list[Path]:
    target = directory or config_dir()
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name in DEFAULT_FILES:
        dest = target / name
        if overwrite or not dest.exists():
            shutil.copyfile(packaged_default(name), dest)
            written.append(dest)

    return written


def load_presets(directory: Path | None = None) -> list[dict]:
    data = load_json(PRESETS_FILE, directory)
    if not isinstance(data, list):
        raise ConfigError(f"{PRESETS_FILE} must contain a list of presets")

    return data


def save_presets(presets: list[dict], directory: Path | None = None) -> Path:
    return save_json(PRESETS_FILE, presets, directory)


def load_led_config(directory: Path | None = None) -> dict:
    data = load_json(LED_CONFIG_FILE, directory)
    if not isinstance(data, dict):
        raise ConfigError(f"{LED_CONFIG_FILE} must contain an object")

    return data


def save_led_config(cfg: dict, directory: Path | None = None) -> Path:
    return save_json(LED_CONFIG_FILE, cfg, directory)


def merge_led_config(current: dict, pol_a_bits: list[int], pol_b_bits: list[int], diffuse_mask: int) -> dict:
    merged = dict(current)
    if list(current.get("pol_a_bits", [])) != list(pol_a_bits):
        merged.pop("light_azimuths_deg", None)
    merged["pol_a_bits"] = list(pol_a_bits)
    merged["pol_b_bits"] = list(pol_b_bits)
    merged["diffuse_mask"] = int(diffuse_mask)

    return merged


def load_state(directory: Path | None = None) -> dict:
    path = (directory or config_dir()) / STATE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}


def update_state(directory: Path | None = None, **values) -> None:
    state = load_state(directory)
    state.update(values)
    try:
        save_json(STATE_FILE, state, directory)
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not write %s: %s", STATE_FILE, exc)


def default_color_calibration_path(directory: Path | None = None) -> Path | None:
    directory = directory or config_dir()
    remembered = load_state(directory).get("color_calibration")
    if remembered and Path(remembered).is_file():
        return Path(remembered)
    candidate = directory / COLOR_CALIBRATION_FILE

    return candidate if candidate.is_file() else None


def setup_logging(directory: Path | None = None, verbose: bool = False) -> Path:
    directory = directory or log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOG_FILE
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        if getattr(handler, "_fabsim_rig", False):
            root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in (file_handler, console):
        handler.setFormatter(fmt)
        handler._fabsim_rig = True
        root.addHandler(handler)

    return path
