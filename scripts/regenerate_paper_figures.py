"""Thin compatibility wrapper around the modern ``python -m subselect`` CLI.

The canonical entry point post-restructure is ``python -m subselect <country>``;
this script is kept for backward compatibility with existing prompts and is
flagged as deprecated. New automation should call the CLI directly.

Usage::

    python scripts/regenerate_paper_figures.py --country greece [--include-seasonal-bias]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from subselect.__main__ import main as cli_main


def main() -> None:
    warnings.warn(
        "scripts/regenerate_paper_figures.py is deprecated; "
        "use 'python -m subselect <country>' instead.",
        DeprecationWarning, stacklevel=2,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="greece")
    parser.add_argument("--include-seasonal-bias", action="store_true")
    args = parser.parse_args()
    forwarded = [args.country]
    if args.include_seasonal_bias:
        forwarded.append("--include-seasonal-bias")
    sys.exit(cli_main(forwarded))


if __name__ == "__main__":
    main()
