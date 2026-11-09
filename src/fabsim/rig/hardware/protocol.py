from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

BAUD = 9600

#: Reply kinds returned by :func:`parse_reply`.
MOVE_DONE = "move_done"
LIMIT_TRIP = "limit_trip"
LIMIT_ARMED = "limit_armed"
LIMIT_OFF = "limit_off"
LEDS_OK = "leds_ok"
LEDS_OFF = "leds_off"
POWER = "power"
TENSION = "tension"
JOG = "jog"
ZERO = "zero"
COEF = "coef"
STATUS = "status"
TRIGGER = "trigger"
BANNER = "banner"
UNKNOWN_COMMAND = "unknown_command"
OTHER = "other"


def jog(steps: int) -> str:
    steps = int(steps)

    return f"+{steps}" if steps >= 0 else f"-{-steps}"


def leds(mask: int) -> str:
    if not 0 <= int(mask) <= 255:
        raise ValueError(f"LED mask must be 0..255, got {mask}")

    return f"L{int(mask)}"


def power(value: int) -> str:
    if not 0 <= int(value) <= 255:
        raise ValueError(f"LED power must be 0..255, got {value}")

    return f"P{int(value)}"


def speed(mult: int) -> str:
    return f"V{max(int(mult), 1)}"


def tare() -> str:
    return "@"


def set_coefficient(newtons_per_count: float) -> str:
    return f"*{newtons_per_count:.8f}"


def read_tension() -> str:
    return "t"


def force_limit(
    force_n: float, release_steps: int = 0, release_dir_positive: bool = False
) -> str:
    if force_n <= 0:
        return "F0"

    return f"F{force_n:.6f},{int(release_steps)},{1 if release_dir_positive else 0}"


@dataclass(frozen=True)
class Reply:

    kind: str
    line: str
    data: dict = field(default_factory=dict)


_FIELD_RE = re.compile(r"([a-z_]+)=([-+0-9.eE]+)")
_ZERO_RE = re.compile(r"^zero:\s*(-?\d+)")


def parse_force_telemetry(line: str) -> dict | None:
    text = str(line).strip()
    if text.startswith("tension:"):
        parts = text.partition(":")[2].strip().split()
        if not parts:
            return None
        try:
            return {
                "force_n": float(parts[0]),
                "raw_counts": None,
                "zero_counts": None,
                "calibration_n_per_count": None,
            }
        except ValueError:
            return None
    if not (text.startswith("TENSION") or text.startswith("LIMIT_TRIP")):
        return None
    fields = dict(_FIELD_RE.findall(text))
    try:
        return {
            "force_n": float(fields["force_n"]),
            "raw_counts": (
                int(float(fields["raw_counts"])) if "raw_counts" in fields else None
            ),
            "zero_counts": (
                int(float(fields["zero_counts"])) if "zero_counts" in fields else None
            ),
            "calibration_n_per_count": (
                float(fields["calibration_n_per_count"])
                if "calibration_n_per_count" in fields
                else None
            ),
        }
    except (KeyError, ValueError):
        return None


def parse_reply(line: str) -> Reply:
    text = line.strip()
    if "MOVE_DONE" in text:
        return Reply(MOVE_DONE, text)
    if text.startswith("LIMIT_TRIP"):
        telemetry = parse_force_telemetry(text)
        if telemetry is None:
            # Older firmware printed ``LIMIT_TRIP <force>``.
            try:
                telemetry = {"force_n": float(text.split()[1])}
            except (IndexError, ValueError):
                telemetry = {}

        return Reply(LIMIT_TRIP, text, telemetry)
    if text.startswith("LIMIT_ARMED"):
        return Reply(
            LIMIT_ARMED, text, {k: float(v) for k, v in _FIELD_RE.findall(text)}
        )
    if text.startswith("limit: off"):
        return Reply(LIMIT_OFF, text)
    if text.startswith("leds:"):
        return Reply(LEDS_OK if "OK" in text else LEDS_OFF, text)
    if text.startswith("power:"):
        return Reply(POWER, text)
    if text.startswith("TENSION") or text.startswith("tension:"):
        return Reply(TENSION, text, parse_force_telemetry(text) or {})
    if text.startswith("jog:"):
        return Reply(JOG, text)
    match = _ZERO_RE.match(text)
    if match:
        return Reply(ZERO, text, {"zero_counts": int(match.group(1))})
    if text.startswith("coef:"):
        return Reply(COEF, text)
    if text.startswith("status:"):
        return Reply(STATUS, text)
    if text == "trigger":
        return Reply(TRIGGER, text)
    if text.endswith("stretch-motor ready"):
        return Reply(BANNER, text)
    if re.fullmatch(r"\?.", text):
        return Reply(UNKNOWN_COMMAND, text)

    return Reply(OTHER, text)


def overload_reason(force_n: float | None, max_force_n: float) -> str | None:
    if max_force_n <= 0 or force_n is None or not math.isfinite(force_n):
        return None
    if abs(force_n) >= max_force_n:
        return f"force_limit_exceeded ({force_n:.3f} N >= {max_force_n:.3f} N)"

    return None
