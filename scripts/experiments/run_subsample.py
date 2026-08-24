"""Subsample robustness and decay analysis.

Part 6: evaluates each survivor across 7 time periods + rolling decay.

Usage
-----
    python scripts/experiments/run_subsample.py \
        --experiment-dir data/experiments/<timestamp>_<run>_<pair>
    python scripts/experiments/run_subsample.py \
        --experiment-dirs data/experiments/<exp-a> data/experiments/<exp-b>
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from siglab.assay.sample import POSTHOC_RESULTS_SEMANTICS

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from tqdm import tqdm

CACHE_DIR  = REPO_ROOT / "data" / "cache"

# This branch changed the VALUES several posthoc scripts compute (grs_p,
# subsample/decay labels, wls_fmb_t). Every payload this script writes
# carries this tag so --resume can detect and refuse to silently mix
# pre-fix rows with post-fix rows in the same results file.
RESULTS_SEMANTICS = POSTHOC_RESULTS_SEMANTICS
DEFAULT_WORKERS = 4
MAX_WORKERS = 6


_SUBSAMPLE_WORKER_DATA: dict | None = None
_THREADPOOL_LIMITER = None


def _experiment_dirs(args) -> list[Path]:
    dirs = []
    if args.experiment_dir:
        dirs.extend(args.experiment_dir)
    if args.experiment_dirs:
        dirs.extend(args.experiment_dirs)
    return [Path(path) for path in dirs]


def _default_output_path(experiment_dirs: list[Path]) -> Path:
    if len(experiment_dirs) == 1:
        return experiment_dirs[0] / "posthoc" / "subsample" / "results.json"
    return experiment_dirs[0].parent / "posthoc_aggregate" / "subsample" / "results.json"


def _read_existing_output(output_path: Path) -> dict:
    empty = {"subsample": [], "decay": []}
    if not output_path.exists():
        return empty
    try:
        payload = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError):
        return empty
    if not isinstance(payload, dict):
        return empty
    return {
        "subsample": list(payload.get("subsample") or []),
        "decay": list(payload.get("decay") or []),
    }


def _resume_semantics_error(output_path: Path) -> str | None:
    """Refusal message if `output_path` predates RESULTS_SEMANTICS, else None.

    --resume accretively merges prior subsample/decay rows into new output
    with no semantics marker of its own, so resuming over a file written
    before this branch's value changes would silently mix old-semantics and
    new-semantics rows in one file. A missing/empty payload has nothing to
    mix and is always fine to resume over.
    """
    if not output_path.exists():
        return None
    try:
        payload = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return (
            f"Cannot resume from {output_path}: the existing output is "
            f"unreadable or invalid JSON ({exc}). Move or delete this file, "
            "then rerun without --resume to start a fresh run."
        )
    if not isinstance(payload, dict):
        return None
    if not (payload.get("subsample") or payload.get("decay")):
        return None
    if payload.get("semantics") == RESULTS_SEMANTICS:
        return None
    return (
        f"{output_path} predates the 2026-07 posthoc semantics fixes "
        f"(payload is missing the {RESULTS_SEMANTICS!r} tag). Resuming "
        "would silently mix old-semantics and new-semantics subsample/decay "
        "rows in the same file. Move or delete this file, then rerun "
        "without --resume to start a fresh, correctly-tagged run."
    )


def _completed_signal_ids(output_path: Path) -> set[str]:
    payload = _read_existing_output(output_path)
    sub_ids = {
        str(row["signal_id"])
        for row in payload["subsample"]
        if isinstance(row, dict) and row.get("signal_id")
    }
    decay_ids = {
        str(row["signal_id"])
        for row in payload["decay"]
        if isinstance(row, dict) and row.get("signal_id")
    }
    return sub_ids & decay_ids


def _pending_survivors(survivors: list[dict], completed_ids: set[str]) -> list[dict]:
    return [
        sig for sig in survivors
        if str(sig.get("signal_id")) not in completed_ids
    ]


def _write_existing_output(output_path: Path, payload: dict, *, default=None) -> None:
    from siglab.utils.json import write_strict_json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(output_path, payload, indent=2)


def _load_common():
    import pandas as pd
    from siglab.data import cache, crsp as crsp_mod
    from siglab.agent.executor import SignalEngine
    from siglab.assay.sample import build_monthly_sample

    cache.set_cache_dir(CACHE_DIR)
    crsp_panels = crsp_mod.get_crsp_panels(use_cache=True)
    crsp_raw = cache.load_panel("crsp_1960_2024_raw")
    crsp_raw["date"]  = pd.to_datetime(crsp_raw["date"])
    crsp_raw["month"] = crsp_raw["date"].dt.month
    crsp_raw["year"]  = crsp_raw["date"].dt.year
    comp_ccm = cache.load_panel("comp_ccm_1960_2024")

    sic_panel, fin_panel = None, None
    if "siccd" in crsp_raw.columns:
        sic_panel = crsp_raw.pivot_table(values="siccd", index="date", columns="permno", aggfunc="first")
        sic_panel.index = pd.to_datetime(sic_panel.index)
        fin_panel = (sic_panel >= 6000) & (sic_panel < 7000)

    engine = SignalEngine(comp=comp_ccm, crsp=crsp_raw, sic_panel=sic_panel)
    sample = build_monthly_sample(
        crsp_panels["ret"], crsp_panels["me"], crsp_panels["nyse"]
    )
    return dict(engine=engine, sample=sample, fin_panel=fin_panel)


def _validate_worker_count(workers: int) -> int:
    workers = int(workers)
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    return workers


def _init_subsample_worker() -> None:
    """Validate inherited data and prevent nested native-thread fan-out."""
    global _THREADPOOL_LIMITER
    if _SUBSAMPLE_WORKER_DATA is None:
        raise RuntimeError("subsample worker started without inherited data")
    from threadpoolctl import threadpool_limits

    _THREADPOOL_LIMITER = threadpool_limits(limits=1)


def _subsample_task(sig: dict) -> tuple:
    """Reduce a survivor row to the small specification needed by a worker."""
    return (
        str(sig["signal_id"]),
        str(sig["name"]),
        str(sig["expression"]),
        str(sig.get("expected_sign", "positive")),
        bool(sig.get("exclude_financials", True)),
        bool(sig.get("exclude_microcap", True)),
    )


def _run_subsample_worker(task: tuple) -> dict:
    """Build and evaluate one signal using process-local heavy data."""
    from siglab.assay.sample import prepare_signal_panel
    from siglab.assay.subsample import run_decay_analysis, run_subsample_tests

    if _SUBSAMPLE_WORKER_DATA is None:
        raise RuntimeError("subsample worker was not initialized")

    (
        signal_id,
        name,
        expression,
        expected_sign,
        exclude_financials,
        exclude_microcap,
    ) = task
    data = _SUBSAMPLE_WORKER_DATA
    try:
        raw = data["engine"].execute(expression)
        prepared = prepare_signal_panel(
            raw,
            data["sample"],
            financials=data.get("fin_panel"),
            exclude_financials=exclude_financials,
            exclude_microcap=exclude_microcap,
        )
    except Exception as exc:
        return {
            "signal_id": signal_id,
            "name": name,
            "error": str(exc),
        }

    sub = run_subsample_tests(
        name,
        prepared.signal,
        data["sample"].returns,
        expected_sign=expected_sign,
    )
    sub_row = asdict(sub)
    sub_row["signal_id"] = signal_id

    decay = run_decay_analysis(name, prepared.signal, data["sample"].returns)
    decay_row = asdict(decay)
    decay_row["signal_id"] = signal_id
    return {
        "signal_id": signal_id,
        "name": name,
        "subsample": sub_row,
        "decay": decay_row,
        "error": None,
    }


def run_subsample_tasks(survivors: list[dict], *, workers: int) -> list[dict]:
    """Run survivor specs in deterministic input order."""
    global _SUBSAMPLE_WORKER_DATA
    workers = _validate_worker_count(workers)
    tasks = [_subsample_task(sig) for sig in survivors]
    if not tasks:
        return []
    data = _load_common()
    _SUBSAMPLE_WORKER_DATA = data
    effective_workers = min(workers, len(tasks))
    try:
        if effective_workers == 1:
            return list(map(_run_subsample_worker, tasks))

        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_subsample_worker,
        ) as pool:
            return list(pool.map(_run_subsample_worker, tasks))
    finally:
        _SUBSAMPLE_WORKER_DATA = None


def _load_survivors(experiment_dirs: list[Path]):
    from siglab.lab.posthoc import collect_horse_race_survivors

    return collect_horse_race_survivors(experiment_dirs)


def _build_panels(survivors, data):
    from siglab.assay.sample import prepare_signal_panel
    panels = {}
    for sig in tqdm(survivors, desc="Building panels", unit="sig"):
        try:
            raw = data["engine"].execute(sig["expression"])
            prepared = prepare_signal_panel(
                raw,
                data["sample"],
                financials=data.get("fin_panel"),
                exclude_financials=bool(sig.get("exclude_financials", True)),
                exclude_microcap=bool(sig.get("exclude_microcap", True)),
            )
            panels[sig["signal_id"]] = prepared.signal
        except Exception as e:
            tqdm.write(f"  Skip {sig['name']}: {e}")
    return panels


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Subsample robustness and decay")
    parser.add_argument("--experiment-dir", action="append")
    parser.add_argument("--experiment-dirs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, MAX_WORKERS + 1),
        default=DEFAULT_WORKERS,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    experiment_dirs = _experiment_dirs(args)
    if not experiment_dirs:
        parser.error("provide --experiment-dir or --experiment-dirs")
    from siglab.lab.posthoc import validate_experiment_dirs

    try:
        experiment_dirs = validate_experiment_dirs(experiment_dirs)
    except ValueError as exc:
        parser.error(str(exc))

    out = Path(args.output) if args.output else _default_output_path(experiment_dirs)
    if args.resume:
        semantics_error = _resume_semantics_error(out)
        if semantics_error:
            parser.error(semantics_error)
    existing = _read_existing_output(out) if args.resume else {"subsample": [], "decay": []}
    survivors = _load_survivors(experiment_dirs)
    if not survivors:
        print("No horse-race survivors found."); return
    completed_ids = _completed_signal_ids(out) if args.resume else set()
    if completed_ids:
        print(f"Resuming subsample+decay: skipping {len(completed_ids)} completed signals")
        survivors = _pending_survivors(survivors, completed_ids)
    if not survivors:
        if args.resume:
            _write_existing_output(out, {"semantics": RESULTS_SEMANTICS, **existing})
            print(f"No pending subsample+decay signals; preserved existing output -> {out}")
        else:
            print("No pending subsample+decay signals.")
        return
    print(f"{len(survivors)} horse-race survivors")

    print(f"\nSubsample + decay analysis...\n")
    sub_results, decay_results = [], []
    results = run_subsample_tasks(survivors, workers=args.workers)
    for item in tqdm(results, desc="Subsample+Decay", unit="sig"):
        if item.get("error"):
            tqdm.write(f"  Skip {item['name']}: {item['error']}")
            continue
        sub_row = item["subsample"]
        decay_row = item["decay"]
        sub_results.append(sub_row)
        decay_results.append(decay_row)

        parts = [f"{k}={v:.2f}" if v is not None else f"{k}=--"
                 for k, v in list(sub_row["results"].items())[:4]]
        tqdm.write(f"  {item['name']:<35s} {' | '.join(parts)}  decay={decay_row['classification']}")

    _write_existing_output(out, {
        "semantics": RESULTS_SEMANTICS,
        "subsample": existing["subsample"] + sub_results,
        "decay": existing["decay"] + decay_results,
    })
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
