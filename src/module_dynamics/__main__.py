"""Top-level entry point for the manuscript comparator models."""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Print a usage hint for the comparator entry point.

    Args:
        argv: Unused argument vector.

    Returns:
        Non-zero exit status requesting the caller to pick a subpackage.
    """

    del argv
    message = (
        "Run the statistical and learned comparators explicitly:\n"
        "  python3 -m src.module_dynamics.basic_rollout {prepare,baseline,train} ...\n"
    )
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
