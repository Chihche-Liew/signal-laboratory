#!/usr/bin/env python
"""Refresh cached Compustat panels after schema changes.

The discovery prompt is backed by ``VARIABLE_CATALOG`` and theme definitions,
while experiments read cached parquet files. This command refreshes the raw
annual Compustat cache and rebuilds the merged Compustat-CCM cache so prompt
variables cannot silently drift away from local data.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from siglab.agent.themes import THEMES
from siglab.agent.variables import VARIABLE_CATALOG
from siglab.data import cache
from siglab.data.compustat import (
    ANNUAL_VARS,
    compute_book_equity,
    download_compustat_annual,
)
from siglab.data.merge import download_ccm_link


DEFAULT_START = "1960-01-01"
DEFAULT_END = "2024-12-31"


@dataclass(frozen=True)
class CacheKeys:
    compustat: str
    ccm: str
    comp_ccm: str


def _year(date_text: str) -> str:
    return pd.Timestamp(date_text).strftime("%Y")


def cache_keys(start: str = DEFAULT_START, end: str = DEFAULT_END) -> CacheKeys:
    start_year = _year(start)
    end_year = _year(end)
    return CacheKeys(
        compustat=f"compustat_annual_{start_year}_{end_year}",
        ccm=f"ccm_link_{start_year}_{end_year}",
        comp_ccm=f"comp_ccm_{start_year}_{end_year}",
    )


def prompt_variable_names() -> set[str]:
    """Variables the prompt can invite the model to use."""
    theme_vars = {var for theme in THEMES.values() for var in theme.variables}
    return set(VARIABLE_CATALOG) | theme_vars


def missing_prompt_columns(df: pd.DataFrame) -> list[str]:
    return sorted(prompt_variable_names() - set(df.columns))


def missing_requested_columns(df: pd.DataFrame) -> list[str]:
    expected = set(ANNUAL_VARS) | {"sic", "se", "dt", "ps", "be"}
    return sorted(expected - set(df.columns))


def prepare_compustat(comp: pd.DataFrame) -> pd.DataFrame:
    """Normalize downloaded annual Compustat for existing cache consumers."""
    prepared = comp.copy()
    prepared["datadate"] = pd.to_datetime(prepared["datadate"])
    if "sich" in prepared.columns:
        prepared["sic"] = prepared["sich"]
    return compute_book_equity(prepared)


def build_comp_ccm(comp: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    """Build the annual Compustat-CCM cache used by strict posthoc suites."""
    merged = comp.merge(
        ccm[["gvkey", "permno", "linkdt", "linkenddt"]],
        on="gvkey",
        how="inner",
    )
    merged = merged[
        (merged["datadate"] >= merged["linkdt"])
        & (merged["datadate"] <= merged["linkenddt"])
    ].copy()
    merged = merged.drop(columns=["linkdt", "linkenddt"])
    merged["year"] = merged["datadate"].dt.year
    merged = merged.sort_values(["permno", "year", "datadate"])
    merged = merged.drop_duplicates(subset=["permno", "year"], keep="last")
    merged["form_year"] = merged["year"] + 1
    return merged


def assert_prompt_schema(df: pd.DataFrame, *, label: str) -> None:
    missing = missing_prompt_columns(df)
    if missing:
        raise RuntimeError(
            f"{label} is missing prompt/catalog variables: {', '.join(missing)}"
        )


def print_schema_report(df: pd.DataFrame, *, label: str) -> None:
    prompt_missing = missing_prompt_columns(df)
    requested_missing = missing_requested_columns(df)
    print(f"{label}: rows={len(df):,} cols={len(df.columns):,}")
    print(f"{label}: missing prompt vars={prompt_missing or '[]'}")
    print(f"{label}: missing requested/cache vars={requested_missing or '[]'}")


def refresh_compustat_cache(
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Path = REPO_ROOT / "data" / "cache",
    refresh_ccm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = cache_keys(start, end)
    cache.set_cache_dir(cache_dir)

    print(f"Downloading Compustat annual {start} to {end}...")
    comp = prepare_compustat(download_compustat_annual(start=start, end=end))
    assert_prompt_schema(comp, label=keys.compustat)
    cache.save_panel(keys.compustat, comp)
    print_schema_report(comp, label=keys.compustat)

    if refresh_ccm or not cache.panel_exists(keys.ccm):
        print(f"Downloading CCM link table {start} to {end}...")
        ccm = download_ccm_link(start=start, end=end)
        cache.save_panel(keys.ccm, ccm)
    else:
        ccm = cache.load_panel(keys.ccm)

    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])

    comp_ccm = build_comp_ccm(comp, ccm)
    assert_prompt_schema(comp_ccm, label=keys.comp_ccm)
    cache.save_panel(keys.comp_ccm, comp_ccm)
    print_schema_report(comp_ccm, label=keys.comp_ccm)
    return comp, comp_ccm


def check_cached_schema(
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Path = REPO_ROOT / "data" / "cache",
) -> int:
    keys = cache_keys(start, end)
    cache.set_cache_dir(cache_dir)
    exit_code = 0
    for key in (keys.compustat, keys.comp_ccm):
        try:
            df = cache.load_panel(key)
        except FileNotFoundError as exc:
            print(str(exc))
            exit_code = 1
            continue
        print_schema_report(df, label=key)
        if missing_prompt_columns(df):
            exit_code = 1
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "cache")
    parser.add_argument(
        "--refresh-ccm",
        action="store_true",
        help="Also re-download the CCM link table instead of reusing the cached one.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report whether cached panels cover prompt/catalog variables.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        return check_cached_schema(
            start=args.start,
            end=args.end,
            cache_dir=args.cache_dir,
        )
    refresh_compustat_cache(
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        refresh_ccm=args.refresh_ccm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
