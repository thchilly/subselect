"""CLI entry: ``python -m subselect <country>``.

Default behaviour computes every L1 artefact for the country (caching to
``cache/<country>/``) and renders every L2 figure to
``results/<country>/figures/``. The whole pipeline runs in 1–3 minutes on a
fresh country, <30 seconds on a cached country.

Flags allow one-shot tuning:

    --no-figures          skip rendering (cache fills only)
    --no-recompute        skip compute, render from cache (fails if empty)
    --only group1,group2  restrict to artefact / figure groups
    --force               ignore cache, recompute everything
    --output-dir PATH     override results/<country>/figures/
    --include-seasonal-bias  render the four seasonal bias maps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from subselect.compute import compute
from subselect.config import Config
from subselect.render import render
from subselect.state import SubselectState


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m subselect",
        description="Compute + render the full country pipeline for one country.",
    )
    parser.add_argument(
        "country", nargs="?", default=None,
        help="Country name (e.g. greece, sweden, portugal). Omit with --global-only.",
    )
    parser.add_argument(
        "--global-only", action="store_true",
        help=(
            "Populate the country-independent global cache (cache/_global/) "
            "without running for any specific country. Useful for cluster "
            "warmup or web-app deployment prep."
        ),
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip rendering; only fill the cache.",
    )
    parser.add_argument(
        "--no-recompute", action="store_true",
        help="Skip compute; render from cache only (fails if cache is empty).",
    )
    parser.add_argument(
        "--only", default=None,
        help=(
            "Comma-separated artefact groups to compute / figure groups to render. "
            "Compute groups: performance, spread, profile. "
            "Figure groups: performance, spread, country_profile."
        ),
    )
    parser.add_argument(
        "--force", default=None, choices=["all", "country", "global"],
        help=(
            "Ignore cache and recompute. 'all' rebuilds both caches, "
            "'country' rebuilds only cache/<country>/, "
            "'global' rebuilds only cache/_global/."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the figure output root.",
    )
    parser.add_argument(
        "--include-seasonal-bias", action="store_true",
        help="Render the four seasonal bias maps in addition to the annual one.",
    )
    parser.add_argument(
        "--no-bias-maps", action="store_true",
        help="Skip the bias-map fields entirely (cheap if you don't need them).",
    )
    return parser.parse_args(argv)


def _load_state_from_cache(country: str, config: Config) -> SubselectState:
    """Run ``compute(force=False)``: every artefact already in cache will be
    cache-hit-loaded; anything missing will be (re)computed. With a populated
    cache this is the <30s fast path."""
    return compute(country, config=config)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m subselect <country> [flags]``.

    Runs the L1 compute pipeline (populating the country cache, and the
    global cache on first use) followed by the L2 figure render. With
    ``--global-only``, builds the global cache without rendering for any
    country.

    Parameters
    ----------
    argv
        Argument vector; defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``2`` on missing-country
        misuse).

    Examples
    --------
    .. code:: bash

        python -m subselect greece
        python -m subselect sweden --no-bias-maps
        python -m subselect --global-only
    """
    args = _parse_args(argv)
    config = Config.from_env()

    only = tuple(s.strip() for s in args.only.split(",")) if args.only else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    force_arg: bool | str = args.force if args.force else False

    t0 = time.time()

    if args.global_only:
        from subselect.compute_global import compute_global
        print("[subselect] global-only run")
        compute_global(config=config, force=force_arg in ("all", "global"))
        print(f"[subselect] global cache populated in {time.time() - t0:.1f} s")
        return 0

    if not args.country:
        print("[subselect] error: country argument required (or use --global-only)")
        return 2

    print(f"[subselect] country={args.country}")

    if args.no_recompute:
        state = _load_state_from_cache(args.country, config)
    else:
        state = compute(
            args.country,
            only=only, force=force_arg, config=config,
            include_bias_maps=not args.no_bias_maps,
            include_seasonal_bias=args.include_seasonal_bias,
        )
    t_compute = time.time() - t0
    print(f"[subselect] L1 compute done in {t_compute:.1f} s")

    if not args.no_figures:
        t1 = time.time()
        figures = render(
            state, country=args.country, output_dir=output_dir,
            only=only, config=config,
            include_seasonal_bias=args.include_seasonal_bias,
        )
        t_render = time.time() - t1
        print(f"[subselect] L2 render: {len(figures)} figures in {t_render:.1f} s")
        for name, path in figures.items():
            print(f"  → {path.relative_to(config.results_root)}")

    print(f"[subselect] total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
