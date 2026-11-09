from __future__ import annotations

import importlib
import inspect
import re
import sys
from collections.abc import Callable

from . import __version__

#: name -> (module, one-line help)
COMMANDS: dict[str, tuple[str, str]] = {
    "package": (
        "fabsim.material_package",
        "build material packages from a measurement project",
    ),
    "ps": (
        "fabsim.ps_frames",
        "write per-run photometric-stereo albedo and normal maps",
    ),
    "colorcal": (
        "fabsim.color_calibration",
        "fit a colour-calibration bundle from chart patches",
    ),
    "validate": ("fabsim.validate", "sanity-check the run folders of a project"),
    "rig": ("fabsim.rig.app", "acquisition application (GUI)"),
    "stereo": ("fabsim.rig.stereo", "photometric-stereo GUI"),
    "ps-verify": ("fabsim.rig.ps_verify", "photometric-stereo verification report"),
}


def _usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: fabsim [--version] <command> [args]", "", "commands:"]
    lines += [f"  {name:<{width}}  {text}" for name, (_, text) in COMMANDS.items()]
    lines += ["", "Run 'fabsim <command> --help' for the options of a command."]

    return "\n".join(lines)


#: Modules whose absence (or a missing shared library behind them) means "optional GUI dependency".
_OPTIONAL_MODULES = ("tkinter", "_tkinter", "serial", "pyueye")
_OPTIONAL_MESSAGE = re.compile(r"tkinter|_tkinter|libtk|libtcl|\bserial\b|pyueye", re.IGNORECASE)


def _is_optional_dependency(exc: ImportError) -> bool:
    if exc.name:
        return exc.name.split(".")[0] in _OPTIONAL_MODULES

    return bool(_OPTIONAL_MESSAGE.search(str(exc)))


def _entry_point(module_name: str, command: str) -> Callable:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if _is_optional_dependency(exc):
            raise SystemExit(
                f"fabsim {command}: missing dependency ({exc})\n"
                "GUI commands need tkinter; the rig also needs the 'rig' extra: pip install 'fabsim[rig]'"
            ) from exc
        raise

    return module.main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_usage())

        return 0 if args else 2
    if args[0] == "--version":
        print(f"fabsim {__version__}")

        return 0
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"fabsim: unknown command '{command}'\n\n{_usage()}", file=sys.stderr)

        return 2
    entry = _entry_point(COMMANDS[command][0], command)
    if inspect.signature(entry).parameters:
        result = entry(rest)
    else:  # an entry point that reads sys.argv itself
        sys.argv = [f"fabsim {command}", *rest]
        result = entry()

    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
