"""Run signal discovery from a RunSpec YAML file."""
from __future__ import annotations

import argparse
import datetime
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

from siglab.lab.evaluator.process_pool import EvaluationProcessPool
from siglab.lab.runner import (
    build_agent_from_env,
    run_experiment,
)
from siglab.lab.run_status import ParallelRunStatus
from siglab.lab.runspec import load_runspecs


DEFAULT_EVAL_WORKERS = 4


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def run_specs_parallel(
    specs: list[Any],
    *,
    agent: Any,
    timestamp: str,
    eval_workers: int = DEFAULT_EVAL_WORKERS,
) -> list[Any]:
    """Run pair-level LLM loops with process-parallel signal evaluation."""
    if not specs:
        return []

    run_status = _build_run_status(specs, timestamp=timestamp)
    if run_status is not None:
        for spec in specs:
            _safe_run_status_update(
                "register pair",
                lambda spec=spec: run_status.register_pair(
                    spec.task.pair,
                    _experiment_dir_for_status(spec, timestamp),
                ),
            )

    max_workers = max(1, len(specs))
    print(
        f"Dispatching {len(specs)} pair jobs on {max_workers} workers...",
        flush=True,
    )
    evaluation_pool = EvaluationProcessPool(
        agent=agent,
        max_workers=eval_workers,
    )

    def run_one(spec):
        log_prefix = f"[{spec.run.name} x {spec.task.pair}]"
        print(f"{log_prefix} starting", flush=True)
        t0 = time.time()
        if run_status is not None:
            _safe_run_status_update(
                "mark pair running",
                lambda: run_status.mark_pair(
                    spec.task.pair,
                    state="running",
                    stage="running",
                ),
            )
        try:
            result = run_experiment(
                spec,
                agent=agent,
                timestamp=timestamp,
                evaluator_wrapper=lambda evaluator: evaluation_pool.wrap(
                    evaluator,
                    pair=spec.task.pair,
                ),
            )
            elapsed = time.time() - t0
            n_sig = len(result.archive.successful(spec.evaluator.success_threshold))
        except Exception as exc:
            elapsed = time.time() - t0
            message = str(exc)
            failure_metadata: dict[str, Any] = {
                "error_type": type(exc).__name__,
                "elapsed_sec": round(elapsed, 3),
            }
            if isinstance(exc, BrokenProcessPool):
                worker_failures = evaluation_pool.broken_worker_failures()
                culprit_pairs = sorted(
                    {
                        row["pair"]
                        for row in worker_failures
                        if row["suspected_culprit"] and row["pair"] is not None
                    }
                )
                crash_context = evaluation_pool.describe_broken_pool(worker_failures)
                if spec.task.pair in culprit_pairs:
                    message = f"Evaluator worker crashed in this pair: {crash_context}"
                else:
                    message = (
                        "Evaluation interrupted by a worker crash in another pair: "
                        f"{crash_context}"
                    )
                failure_metadata.update(
                    {
                        "culprit_pairs": culprit_pairs,
                        "worker_failures": worker_failures,
                    }
                )
            if run_status is not None:
                _safe_run_status_update(
                    "mark pair failed",
                    lambda: run_status.mark_pair(
                        spec.task.pair,
                        state="failed",
                        stage="failed",
                        message=message,
                        metadata=failure_metadata,
                    ),
                )
            raise
        if run_status is not None:
            _safe_run_status_update(
                "mark pair completed",
                lambda: run_status.mark_pair(
                    spec.task.pair,
                    state="completed",
                    stage="completed",
                    generation=getattr(result.archive, "generation", None),
                    metadata={
                        "output_dir": str(result.output_dir),
                        "n_evaluated": len(result.archive.evaluated),
                        "n_successful": n_sig,
                        "elapsed_sec": round(elapsed, 3),
                    },
                ),
            )
        print(
            f"{log_prefix} DONE after {elapsed:.0f}s: "
            f"{len(result.archive.evaluated)} evaluated, {n_sig} significant",
            flush=True,
        )
        print(f"{log_prefix} Experiment folder: {result.output_dir}", flush=True)
        return result

    results = []
    # Entering the process pool starts every evaluator worker before any
    # pair-level LLM thread exists.  This is required for safe copy-on-write
    # inheritance of the large pandas panels on the fork path.
    with evaluation_pool:
        print(
            f"Evaluation process pool: {eval_workers} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_one, spec) for spec in specs]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except BrokenProcessPool:
                    for pending in futures:
                        pending.cancel()
                    print(
                        "FATAL: evaluation process pool broke; "
                        f"{evaluation_pool.describe_broken_pool()}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
    return results


def _build_run_status(specs: list[Any], *, timestamp: str) -> ParallelRunStatus | None:
    first = specs[0]
    try:
        return ParallelRunStatus(
            output_root=Path(first.run.output_root),
            run_name=first.run.name,
            timestamp=timestamp,
        )
    except Exception as exc:
        print(
            f"WARNING: aggregate run status unavailable: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


def _safe_run_status_update(action: str, update: Callable[[], None]) -> None:
    try:
        update()
    except Exception as exc:
        print(
            f"WARNING: aggregate run status {action} failed: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _experiment_dir_for_status(spec: Any, timestamp: str) -> Path:
    if callable(getattr(spec, "experiment_folder", None)):
        return Path(spec.experiment_folder(timestamp))
    return Path(spec.run.output_root) / f"{timestamp}_{spec.run.name}_{spec.task.pair}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run signal discovery from RunSpec YAML")
    parser.add_argument("--config", required=True, help="Path to RunSpec YAML")
    parser.add_argument("--pair", help="Override task pair list with one pair")
    parser.add_argument("--gens", type=int, help="Override loop.n_generations")
    parser.add_argument("--output-dir", help="Override run.output_root")
    parser.add_argument(
        "--eval-workers",
        type=_positive_int,
        default=DEFAULT_EVAL_WORKERS,
        help=(
            "Process workers for CPU-bound signal evaluation "
            f"(default: {DEFAULT_EVAL_WORKERS})"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["default", "guided", "repair", "explore", "exploit"],
        default=None,
    )
    parser.add_argument("--seed-prompt", help="Optional human guidance prompt")
    parser.add_argument(
        "--guidance-scope",
        choices=["none", "task", "loop", "task_and_loop"],
        default=None,
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    specs = load_runspecs(
        args.config,
        pair_override=args.pair,
        gens_override=args.gens,
        output_root_override=args.output_dir,
        mode_override=args.mode,
        seed_prompt=args.seed_prompt,
        guidance_scope=args.guidance_scope,
    )

    agent = build_agent_from_env()
    run_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    print(f"Run timestamp: {run_ts}")

    run_specs_parallel(
        specs,
        agent=agent,
        timestamp=run_ts,
        eval_workers=args.eval_workers,
    )


if __name__ == "__main__":
    main()
