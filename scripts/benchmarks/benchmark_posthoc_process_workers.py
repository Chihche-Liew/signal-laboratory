"""Small 1-vs-4-worker benchmark for post-hoc beta-matrix construction.

This uses deterministic synthetic panels and writes no artifacts. It exercises
the production fork-inherited worker initializer, BLAS pinning, and small
signal-spec tasks used by compact double-bootstrap beta construction.

Usage
-----
    PYTHONPATH=src python scripts/benchmarks/benchmark_posthoc_process_workers.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.experiments import run_double_bootstrap as runner


class SyntheticEngine:
    def __init__(self, base: pd.DataFrame, modifier: pd.DataFrame):
        self.base = base
        self.modifier = modifier

    def execute(self, expression: str) -> pd.DataFrame:
        signal_index = int(expression.rsplit("_", 1)[1])
        return (
            self.base * (1.0 + 0.01 * (signal_index % 11))
            + self.modifier * (0.002 * (signal_index % 7))
        )


class SyntheticAgent:
    def __init__(
        self,
        *,
        signal: pd.DataFrame,
        modifier: pd.DataFrame,
        returns: pd.DataFrame,
    ):
        self.engine = SyntheticEngine(signal, modifier)
        self.ret_panel = returns
        self.me_panel = pd.DataFrame(
            100.0,
            index=returns.index,
            columns=returns.columns,
        )
        self.nyse_panel = pd.DataFrame(
            True,
            index=returns.index,
            columns=returns.columns,
        )
        self.fin_panel = pd.DataFrame(
            False,
            index=returns.index,
            columns=returns.columns,
        )
        self.start_date = returns.index[0]


def _synthetic_inputs(
    *,
    n_signals: int,
    n_months: int,
    n_stocks: int,
    seed: int,
) -> tuple[SyntheticAgent, list[dict]]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1980-01-31", periods=n_months, freq="ME")
    columns = np.arange(10_000, 10_000 + n_stocks)
    signal = pd.DataFrame(
        rng.standard_normal((n_months, n_stocks)),
        index=dates,
        columns=columns,
    )
    modifier = pd.DataFrame(
        rng.standard_normal((n_months, n_stocks)),
        index=dates,
        columns=columns,
    )
    noise = rng.standard_normal((n_months, n_stocks)) * 0.05
    returns = pd.DataFrame(
        noise,
        index=dates,
        columns=columns,
    )
    returns.iloc[1:] += 0.006 * signal.iloc[:-1].to_numpy()
    agent = SyntheticAgent(
        signal=signal,
        modifier=modifier,
        returns=returns,
    )
    specs = [
        {
            "name": f"synthetic_{idx}",
            "expression": f"signal_{idx}",
            "expected_sign": "positive",
            "exclude_financials": False,
            "exclude_microcap": False,
        }
        for idx in range(n_signals)
    ]
    return agent, specs


def _timed_build(
    *,
    agent: SyntheticAgent,
    specs: list[dict],
    context,
    workers: int,
):
    started = perf_counter()
    if workers == 1:
        dataset = runner.build_beta_matrix(
            agent=agent,
            signal_specs=specs,
            progress_path=None,
            nw_lags=0,
            workers=1,
            context=context,
        )
    else:
        dataset = runner.build_beta_matrix(
            agent=agent,
            signal_specs=specs,
            progress_path=None,
            nw_lags=0,
            workers=workers,
            context=context,
        )
    return perf_counter() - started, dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark compact beta construction with 1 and 4 workers",
    )
    parser.add_argument("--signals", type=int, default=256)
    parser.add_argument("--months", type=int, default=360)
    parser.add_argument("--stocks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.signals < 4 or args.months < 5 or args.stocks < 4:
        raise SystemExit("benchmark requires at least 4 signals, 5 months, and 4 stocks")

    agent, specs = _synthetic_inputs(
        n_signals=args.signals,
        n_months=args.months,
        n_stocks=args.stocks,
        seed=args.seed,
    )
    context = runner.beta_build_context_for_agent(
        agent=agent,
        signal_specs=specs,
        nw_lags=0,
    )
    one_seconds, one = _timed_build(
        agent=agent,
        specs=specs,
        context=context,
        workers=1,
    )
    four_seconds, four = _timed_build(
        agent=agent,
        specs=specs,
        context=context,
        workers=4,
    )

    np.testing.assert_array_equal(four.matrix, one.matrix)
    np.testing.assert_array_equal(four.signs, one.signs)
    np.testing.assert_array_equal(four.observed_t, one.observed_t)
    if four.signal_specs != one.signal_specs or four.dates != one.dates:
        raise AssertionError("1-worker and 4-worker output ordering differs")

    print(json.dumps({
        "signals": args.signals,
        "months": args.months,
        "stocks": args.stocks,
        "one_worker_seconds": round(one_seconds, 3),
        "four_worker_seconds": round(four_seconds, 3),
        "speedup": round(one_seconds / four_seconds, 3),
        "outputs_identical": True,
    }, indent=2))


if __name__ == "__main__":
    main()
