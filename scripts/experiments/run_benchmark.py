"""Benchmark harness — run resolved RunSpecs in batch and collate.

Usage
-----
    # Run every RunSpec expanded by the default architecture YAML files
    python scripts/experiments/run_benchmark.py

    # Custom self-contained experiment specs
    python scripts/experiments/run_benchmark.py \
        --configs paper/configs/baseline.yaml paper/configs/ablation_no_critique.yaml

    # Explicit pair override / cross product
    python scripts/experiments/run_benchmark.py \
        --configs paper/configs/baseline.yaml paper/configs/ablation_no_critique.yaml \
        --pairs profit_invest invest_accrual

    # Limit generations for a smoke run
    python scripts/experiments/run_benchmark.py --gens 2

Output: a single JSON file summarizing every (architecture × pair) run, plus
a console comparison table.

All (architecture × pair) jobs are dispatched in parallel via a
ThreadPoolExecutor. A single SignalEvaluator is built once and shared read-only
across every worker; mutable per-run lab state lives in the components created
for each job. Each job builds its own Proposer / Reflector / Stopping / Budget /
Recorder via build_components so per-job state (e.g.
ThinkingChainReflector._prior_turns) does not bleed across runs.
"""
import argparse
import datetime
import json
import sys
import threading
import time
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from siglab.lab.benchmark import BenchmarkMetrics, collect_metrics
from siglab.lab.runner import (
    build_agent_from_env,
    run_experiment,
)
from siglab.lab.runspec import load_runspecs


DEFAULT_CONFIGS = [
    "paper/configs/baseline.yaml",
]


def _artifact_index(experiment_dir: Path) -> dict[str, str]:
    """Return canonical per-run artifact paths for a benchmark summary row."""
    return {
        "experiment_dir": str(experiment_dir),
        "archive_path": str(experiment_dir / "archive.json"),
        "manifest_path": str(experiment_dir / "manifest.json"),
    }


def load_benchmark_jobs(
    config_paths: list[str],
    *,
    pair_overrides: list[str] | None = None,
    gens_override: int | None = None,
) -> list[tuple[str, object]]:
    """Load benchmark jobs from self-contained RunSpec YAML files.

    By default each YAML's own `task.pair` / `task.pairs` is authoritative.
    Passing `pair_overrides` intentionally creates a config x pair cross
    product for smoke tests or targeted comparisons.
    """
    jobs = []
    for cfg_path in config_paths:
        if pair_overrides:
            for pair_name in pair_overrides:
                spec = load_runspecs(
                    cfg_path,
                    pair_override=pair_name,
                    gens_override=gens_override,
                )[0]
                jobs.append((cfg_path, spec))
        else:
            for spec in load_runspecs(cfg_path, gens_override=gens_override):
                jobs.append((cfg_path, spec))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run architecture benchmark")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            "Optional explicit pair override. When omitted, each config's "
            "task.pair/task.pairs is used as written."
        ),
    )
    parser.add_argument("--gens", type=int,
                        help="Override n_generations in every config (for smoke runs)")
    parser.add_argument("--output",
                        default=str(REPO_ROOT / "data" / "experiments" / "benchmark_results.json"))
    args = parser.parse_args()

    # Build the (expensive) SignalEvaluator once and share it across all workers.
    # SignalEvaluator.evaluate_signal only reads self.engine / ret_panel / me_panel /
    # ff_factors / nyse_panel / fin_panel; it has no evaluator-side archive.
    agent = build_agent_from_env()

    run_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    print(f"Run timestamp: {run_ts}")

    # Enumerate jobs up front so we can size the pool. The default path treats
    # each YAML as a self-contained experiment spec; --pairs is an explicit
    # cross-product override.
    jobs = load_benchmark_jobs(
        args.configs,
        pair_overrides=args.pairs,
        gens_override=args.gens,
    )

    max_workers = max(1, len(jobs))
    print(f"Dispatching {len(jobs)} jobs on {max_workers} workers...", flush=True)

    # The evaluator (SignalEvaluator.evaluate_signal) is CPU-bound pandas work.
    # Under N-way ThreadPool contention it goes GIL-serialized AND pays cache
    # thrashing overhead, stalling for so long that the benchmark appears
    # silently frozen. We serialize the evaluate step with a shared lock so
    # LLM calls (I/O-bound, Anthropic httpx is thread-safe) stay concurrent
    # across archs while the pandas-heavy step runs one-at-a-time.
    eval_lock = threading.Lock()

    class _LockedEvaluator:
        """Serializes .evaluate() across worker threads via a shared lock."""
        def __init__(self, inner, lock):
            self._inner = inner
            self._lock = lock

        def evaluate(self, proposals, generation):
            with self._lock:
                return self._inner.evaluate(proposals, generation=generation)

    def run_one(cfg_path: str, spec) -> tuple[BenchmarkMetrics, dict[str, str]]:
        """Run one (arch × pair) DiscoveryLoop. Returns its BenchmarkMetrics."""
        arch_name = spec.run.name
        pair_name = spec.task.pair
        log_prefix = f"[{arch_name} × {pair_name}]"
        print(f"{log_prefix} starting", flush=True)

        t0 = time.time()
        try:
            result = run_experiment(
                spec,
                agent=agent,
                timestamp=run_ts,
                evaluator_wrapper=lambda evaluator: _LockedEvaluator(
                    evaluator, eval_lock,
                ),
            )
            archive = result.archive
            loop = result.loop
            elapsed = time.time() - t0
            stop_decision = loop.stopping.should_stop(
                archive,
                meta={
                    "generation": archive.generation,
                    "task_id": result.task.task_id,
                },
            )
            metrics = collect_metrics(
                architecture=arch_name, pair=pair_name, archive=archive,
                total_input_tokens=loop.budget.total_tokens_in,
                total_output_tokens=loop.budget.total_tokens_out,
                cached_input_tokens=loop.budget.total_cached_in,
                wall_clock_sec=elapsed,
                stopping_reason=stop_decision.reason,
                stopping_generation=archive.generation,
            )
            print(f"{log_prefix} DONE after {elapsed:.0f}s: "
                  f"{metrics.n_evaluated} eval, "
                  f"{metrics.n_raw_fmb_successes} raw FMB successes",
                  flush=True)
            return metrics, _artifact_index(result.output_dir)
        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            tb = traceback.format_exc()
            print(f"{log_prefix} FAILED after {elapsed:.0f}s: {e}\n{tb}", flush=True)
            output_dir = Path(spec.experiment_folder(run_ts))
            return (
                BenchmarkMetrics(
                    architecture=arch_name, pair=pair_name,
                    wall_clock_sec=elapsed,
                    stopping_reason=f"ERROR: {e!r}",
                ),
                _artifact_index(output_dir),
            )

    records: list[tuple[BenchmarkMetrics, dict[str, str]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one, cfg_path, spec) for (cfg_path, spec) in jobs]
        for fut in as_completed(futures):
            records.append(fut.result())

    # Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {**metrics.to_dict(), **artifact_index}
        for metrics, artifact_index in records
    ]
    out_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"\n-> wrote {out_path}")

    # Print comparison table (stable ordering: arch then pair).
    records.sort(key=lambda row: (row[0].architecture, row[0].pair))
    print("\n" + "=" * 80)
    print(f"  {'arch':<20s} {'pair':<20s} {'eval':>5s} {'raw+':>5s} "
          f"{'tokens_in':>10s} {'wall s':>8s}")
    print("  " + "-" * 78)
    for r, _artifact_index in records:
        print(f"  {r.architecture:<20s} {r.pair:<20s} "
              f"{r.n_evaluated:>5d} {r.n_raw_fmb_successes:>5d} "
              f"{r.total_input_tokens:>10d} {r.wall_clock_sec:>8.0f}")

    print(f"\nPer-run artifacts are written under each config's output root as "
          f"{run_ts}_<architecture>_<pair>/")


if __name__ == "__main__":
    main()
