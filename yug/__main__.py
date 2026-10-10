"""YUG CLI entry point.

Phase 0 scope: only ``--version`` is implemented. Subcommands like
``chat`` or ``config validate`` belong to later phases (Phase 6 CLI
integration, per the Directive) once there is a system behind them to
invoke.
"""

from __future__ import annotations

import argparse
import sys

from yug import __milestone__, __version__
from yug.utils.types import VersionInfo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yug",
        description="YUG - Personal Agentic AI System (Phase 0: Foundation)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the YUG version and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        info = VersionInfo(version=__version__, milestone=__milestone__)
        print(info)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
