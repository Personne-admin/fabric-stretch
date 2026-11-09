from __future__ import annotations

import argparse
import logging
import sys

from . import config

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fabsim rig", description="Fabric stretching rig acquisition app")
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="copy the packaged default config files into the user config directory and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --init-config: overwrite existing files",
    )
    parser.add_argument(
        "--paths",
        action="store_true",
        help="print the config and log directories and exit",
    )
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-v", "--verbose", action="store_true", help="log DEBUG messages to the console")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.paths:
        print(f"config: {config.config_dir()}\nlogs:   {config.log_dir()}")

        return 0
    if args.init_config:
        written = config.seed_defaults(overwrite=args.force)
        for path in written:
            print(f"wrote {path}")
        if not written:
            print(f"nothing to do: all default files already exist in {config.config_dir()}")

        return 0

    log_file = config.setup_logging(verbose=args.verbose)
    log.info("Starting rig app; config %s, log %s", config.config_dir(), log_file)
    try:
        from .ui.main_window import App
    except ImportError as exc:
        log.exception("GUI dependencies are missing")
        print(
            f"The rig GUI needs tkinter and the 'rig' extra (pip install fabsim[rig]): {exc}",
            file=sys.stderr,
        )

        return 1
    app = App()
    if args.smoke_test:
        app.after(200, app.destroy)
    app.mainloop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
