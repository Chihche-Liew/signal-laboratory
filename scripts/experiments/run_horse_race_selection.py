"""Run horse-race selection for first-pass discoveries.

Usage
-----
    python scripts/experiments/run_horse_race_selection.py \
        --experiment-dir data/experiments/<timestamp>_<run>_<pair>
    python scripts/experiments/run_horse_race_selection.py \
        --experiment-dirs data/experiments/<exp-a> data/experiments/<exp-b>
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from siglab.assay.sample import POSTHOC_RESULTS_SEMANTICS

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

CACHE_DIR = REPO_ROOT / "data" / "cache"
SCHEMA_VERSION = "3"
RESULTS_SEMANTICS = POSTHOC_RESULTS_SEMANTICS
DEFAULT_WORKERS = 4
MAX_WORKERS = 6


_SELECTION_WORKER_DATA: dict | None = None
_THREADPOOL_LIMITER = None


def _experiment_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = []
    if args.experiment_dir:
        dirs.extend(args.experiment_dir)
    if args.experiment_dirs:
        dirs.extend(args.experiment_dirs)
    return [Path(path) for path in dirs]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_selection_artifact(experiment_dir: Path, payload: dict) -> Path:
    from siglab.utils.json import write_strict_json

    out = _selection_artifact_path(experiment_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(out, payload, indent=2)
    return out


def _selection_artifact_path(experiment_dir: Path) -> Path:
    return experiment_dir / "selection" / "horse_race.json"


def _selection_artifact_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("semantics") == RESULTS_SEMANTICS
    )


def _base_payload(*, experiment_dir: Path, threshold: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "semantics": RESULTS_SEMANTICS,
        "created_at": _utc_now(),
        "selection": "horse_race",
        "input_population": "first_pass_discoveries",
        "threshold": float(threshold),
        "experiment_dir": str(experiment_dir),
    }


def _corr_matrix_payload(corr_matrix) -> dict:
    if hasattr(corr_matrix, "to_dict"):
        return corr_matrix.to_dict()
    return dict(corr_matrix)


def _dropped_row(
    candidate: dict,
    *,
    reason: str,
    fmb_conditional_t=None,
    error: str | None = None,
) -> dict:
    row = {
        "signal_id": candidate["signal_id"],
        "name": candidate.get("name"),
        "expression": candidate.get("expression"),
        "reason": reason,
    }
    if fmb_conditional_t is not None:
        row["fmb_conditional_t"] = fmb_conditional_t
    if error is not None:
        row["error"] = error
    return row


def build_selection_payload(
    *,
    experiment_dir: Path,
    candidates: list[dict],
    result,
    threshold: float,
    extra_dropped: list[dict] | None = None,
) -> dict:
    from siglab.assay.sample import direction_matches

    expected_signs = {
        candidate["signal_id"]: candidate.get("expected_sign", "positive")
        for candidate in candidates
    }
    survivor_ids = list(result.stepwise_survivors)
    survivor_set = set(survivor_ids)
    fmb_conditional_t = dict(result.fmb_conditional_t)
    dropped_reasons = dict(getattr(result, "dropped_reasons", {}))
    extra_dropped = list(extra_dropped or [])
    extra_dropped_ids = {row.get("signal_id") for row in extra_dropped}

    dropped = list(extra_dropped)
    for candidate in candidates:
        signal_id = candidate["signal_id"]
        if signal_id in survivor_set or signal_id in extra_dropped_ids:
            continue
        conditional_t = fmb_conditional_t.get(signal_id)
        reason = dropped_reasons.get(signal_id)
        if reason is None and not _is_finite_number(conditional_t):
            reason = "conditional_t_not_estimable"
        if reason is None and not direction_matches(
            conditional_t,
            expected_signs[signal_id],
        ):
            reason = "conditional_sign_mismatch"
        if reason is None:
            reason = "conditional_t_below_threshold"
        dropped.append(
            _dropped_row(
                candidate,
                reason=reason,
                fmb_conditional_t=conditional_t,
            )
        )

    status = getattr(result, "status", None)
    if status is None:
        all_t_values = [
            *fmb_conditional_t.values(),
            *dict(result.stepwise_t).values(),
        ]
        status = (
            "ok"
            if any(_is_finite_number(value) for value in all_t_values)
            else "not_estimable"
        )
    status_reason = getattr(result, "reason", None)
    if status == "not_estimable" and status_reason is None:
        status_reason = "non_finite_conditional_t"

    return {
        **_base_payload(experiment_dir=experiment_dir, threshold=threshold),
        "n_input": len(candidates),
        "n_survivors": len(survivor_ids),
        "input_signals": [dict(candidate) for candidate in candidates],
        "stepwise_survivors": survivor_ids,
        "stepwise_t": {
            signal_id: result.stepwise_t[signal_id]
            for signal_id in survivor_ids
        },
        "fmb_conditional_t": fmb_conditional_t,
        "fmb_survivors": list(result.fmb_survivors),
        "absolute_fmb_survivors": list(
            getattr(result, "absolute_fmb_survivors", result.fmb_survivors)
        ),
        "corr_matrix": _corr_matrix_payload(result.corr_matrix),
        "dropped": dropped,
        "status": status,
        "reason": status_reason,
    }


def build_empty_selection_payload(
    *,
    experiment_dir: Path,
    threshold: float,
    reason: str,
    candidates: list[dict] | None = None,
    dropped: list[dict] | None = None,
) -> dict:
    candidates = list(candidates or [])
    return {
        **_base_payload(experiment_dir=experiment_dir, threshold=threshold),
        "n_input": len(candidates),
        "n_survivors": 0,
        "input_signals": [dict(candidate) for candidate in candidates],
        "stepwise_survivors": [],
        "stepwise_t": {},
        "fmb_conditional_t": {},
        "fmb_survivors": [],
        "absolute_fmb_survivors": [],
        "corr_matrix": {},
        "dropped": list(dropped or []),
        "reason": reason,
        "status": "not_estimable",
    }


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_empty_selection_artifact(
    *,
    experiment_dir: Path,
    threshold: float,
    reason: str,
    candidates: list[dict] | None = None,
    dropped: list[dict] | None = None,
) -> Path:
    payload = build_empty_selection_payload(
        experiment_dir=experiment_dir,
        threshold=threshold,
        reason=reason,
        candidates=candidates,
        dropped=dropped,
    )
    return _write_selection_artifact(experiment_dir, payload)


def validate_single_sample_filter_group(candidates: list[dict]) -> tuple[bool, bool]:
    flags = {
        (
            bool(candidate.get("exclude_financials", True)),
            bool(candidate.get("exclude_microcap", True)),
        )
        for candidate in candidates
    }
    if len(flags) > 1:
        raise ValueError(
            "horse-race selection requires one sample-filter group per experiment; "
            "mixed exclude_financials/exclude_microcap settings found. Run "
            "separately by task/sample-filter group."
        )
    return next(iter(flags)) if flags else (True, True)


def load_common_data() -> dict:
    import pandas as pd
    from siglab.agent.executor import SignalEngine
    from siglab.data import cache
    from siglab.data import crsp as crsp_mod
    from siglab.assay.sample import build_monthly_sample

    cache.set_cache_dir(CACHE_DIR)

    print("Loading CRSP panels...")
    crsp_panels = crsp_mod.get_crsp_panels(use_cache=True)
    sample = build_monthly_sample(
        crsp_panels["ret"],
        crsp_panels["me"],
        crsp_panels["nyse"],
    )

    print("Loading Compustat-CCM...")
    crsp_raw = cache.load_panel("crsp_1960_2024_raw")
    crsp_raw["date"] = pd.to_datetime(crsp_raw["date"])
    crsp_raw["month"] = crsp_raw["date"].dt.month
    crsp_raw["year"] = crsp_raw["date"].dt.year
    comp_ccm = cache.load_panel("comp_ccm_1960_2024")

    sic_panel, fin_panel = None, None
    if "siccd" in crsp_raw.columns:
        sic_panel = crsp_raw.pivot_table(
            values="siccd", index="date", columns="permno", aggfunc="first"
        )
        sic_panel.index = pd.to_datetime(sic_panel.index)
        fin_panel = (sic_panel >= 6000) & (sic_panel < 7000)

    engine = SignalEngine(comp=comp_ccm, crsp=crsp_raw, sic_panel=sic_panel)
    return {
        "engine": engine,
        "sample": sample,
        "fin_panel": fin_panel,
    }


def _validate_worker_count(workers: int) -> int:
    workers = int(workers)
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    return workers


def _init_selection_worker() -> None:
    """Validate inherited data and prevent nested native-thread fan-out."""
    global _THREADPOOL_LIMITER
    if _SELECTION_WORKER_DATA is None:
        raise RuntimeError("selection worker started without inherited data")
    from threadpoolctl import threadpool_limits

    _THREADPOOL_LIMITER = threadpool_limits(limits=1)


def _run_selection_worker(task: tuple[str, float]) -> dict:
    """Build one payload from inherited data without writing artifacts."""
    experiment_dir, threshold = task
    if _SELECTION_WORKER_DATA is None:
        raise RuntimeError("selection worker was not initialized")
    return build_selection_for_experiment_payload(
        experiment_dir=Path(experiment_dir),
        threshold=threshold,
        data=_SELECTION_WORKER_DATA,
    )


def build_signal_panels(candidates: list[dict], data: dict) -> tuple[dict, list[dict]]:
    from siglab.assay.sample import prepare_signal_panel

    panels = {}
    dropped = []
    for candidate in candidates:
        signal_id = candidate["signal_id"]
        try:
            raw = data["engine"].execute(candidate["expression"])
            prepared = prepare_signal_panel(
                raw,
                data["sample"],
                financials=data.get("fin_panel"),
                exclude_financials=candidate.get("exclude_financials", True),
                exclude_microcap=candidate.get("exclude_microcap", True),
            )
            panels[signal_id] = prepared.signal
        except Exception as exc:
            dropped.append(
                _dropped_row(
                    candidate,
                    reason="panel_build_error",
                    error=str(exc),
                )
            )
            print(f"  Skip {candidate.get('name', signal_id)}: {exc}")
    return panels, dropped


def build_selection_for_experiment_payload(
    *,
    experiment_dir: Path,
    threshold: float,
    data: dict | None = None,
) -> dict:
    from siglab.assay.horse_race import horse_race
    from siglab.lab.posthoc import collect_first_pass_discoveries

    candidates = collect_first_pass_discoveries([experiment_dir], threshold=threshold)
    if not candidates:
        return build_empty_selection_payload(
            experiment_dir=experiment_dir,
            threshold=threshold,
            reason="no_first_pass_discoveries",
        )

    validate_single_sample_filter_group(candidates)

    if data is None:
        data = load_common_data()

    panels, panel_drops = build_signal_panels(candidates, data)
    if not panels:
        return build_empty_selection_payload(
            experiment_dir=experiment_dir,
            threshold=threshold,
            reason="no_runnable_first_pass_discoveries",
            candidates=candidates,
            dropped=panel_drops,
        )

    result = horse_race(
        panels,
        data["sample"].returns,
        me_panel=None,
        t_threshold=threshold,
        expected_signs={
            candidate["signal_id"]: candidate.get("expected_sign", "positive")
            for candidate in candidates
            if candidate["signal_id"] in panels
        },
    )
    return build_selection_payload(
        experiment_dir=experiment_dir,
        candidates=candidates,
        result=result,
        threshold=threshold,
        extra_dropped=panel_drops,
    )


def run_selection_for_experiment(
    *,
    experiment_dir: Path,
    threshold: float,
    data: dict | None = None,
) -> Path:
    payload = build_selection_for_experiment_payload(
        experiment_dir=experiment_dir,
        threshold=threshold,
        data=data,
    )
    return _write_selection_artifact(experiment_dir, payload)


def run_selection_for_experiments(
    experiment_dirs: list[Path],
    *,
    threshold: float,
    workers: int,
    skip_existing: bool,
) -> list[Path]:
    global _SELECTION_WORKER_DATA
    workers = _validate_worker_count(workers)
    pending = []
    output_by_experiment: dict[Path, Path] = {}
    for exp_dir in experiment_dirs:
        artifact = _selection_artifact_path(exp_dir)
        if skip_existing and _selection_artifact_is_current(artifact):
            output_by_experiment[exp_dir] = artifact
        else:
            pending.append(exp_dir)

    if not pending:
        return [output_by_experiment[exp_dir] for exp_dir in experiment_dirs]

    data = load_common_data()
    effective_workers = min(workers, len(pending))
    if effective_workers == 1:
        for exp_dir in pending:
            payload = build_selection_for_experiment_payload(
                experiment_dir=exp_dir,
                threshold=threshold,
                data=data,
            )
            output_by_experiment[exp_dir] = _write_selection_artifact(
                exp_dir,
                payload,
            )
        return [output_by_experiment[exp_dir] for exp_dir in experiment_dirs]

    tasks = [(str(exp_dir), float(threshold)) for exp_dir in pending]
    _SELECTION_WORKER_DATA = data
    try:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_selection_worker,
        ) as pool:
            payloads = pool.map(_run_selection_worker, tasks)
            for exp_dir, payload in zip(pending, payloads):
                output_by_experiment[exp_dir] = _write_selection_artifact(
                    exp_dir,
                    payload,
                )
    finally:
        _SELECTION_WORKER_DATA = None
    return [output_by_experiment[exp_dir] for exp_dir in experiment_dirs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Horse-race first-pass discoveries")
    parser.add_argument("--experiment-dir", action="append")
    parser.add_argument("--experiment-dirs", nargs="+")
    parser.add_argument("--threshold", type=float, default=1.96)
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, MAX_WORKERS + 1),
        default=DEFAULT_WORKERS,
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    experiment_dirs = _experiment_dirs(args)
    if not experiment_dirs:
        parser.error("provide --experiment-dir or --experiment-dirs")

    from siglab.lab.posthoc import validate_experiment_dirs

    try:
        experiment_dirs = validate_experiment_dirs(experiment_dirs)
    except ValueError as exc:
        parser.error(str(exc))

    outputs = run_selection_for_experiments(
        experiment_dirs,
        threshold=args.threshold,
        workers=args.workers,
        skip_existing=args.skip_existing,
    )
    for out in outputs:
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
