"""Harvey-Liu double-bootstrap from completed experiment folders."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from siglab.assay.sample import POSTHOC_RESULTS_SEMANTICS

from dotenv import load_dotenv

from siglab.lab.posthoc import (
    collect_evaluated_signals,
    collect_first_pass_discoveries,
    validate_experiment_dirs,
)


CACHE_SCHEMA_VERSION = "4"

# Effective --nw-lags when the flag is omitted. The argparse default is a
# None sentinel so "user explicitly passed --nw-lags" is detectable — the
# legacy dense-panel path must reject it (it hardcodes nw_lags=0).
NW_LAGS_DEFAULT = 6
DEFAULT_BETA_WORKERS = 4
MAX_BETA_WORKERS = 6


@dataclass
class BetaMatrixDataset:
    matrix: np.ndarray
    dates: list[str]
    signal_specs: list[dict]
    signs: np.ndarray
    observed_t: np.ndarray
    spec_hash: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class BetaBuildContext:
    dates: list[str]
    metadata: dict[str, object]
    sample: object | None = field(default=None, repr=False, compare=False)


_BETA_WORKER_STATE: dict | None = None
_THREADPOOL_LIMITER = None


def _experiment_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = []
    if args.experiment_dir:
        dirs.extend(args.experiment_dir)
    if args.experiment_dirs:
        dirs.extend(args.experiment_dirs)
    return [Path(path) for path in dirs]


def _default_output_path(experiment_dirs: list[Path]) -> Path:
    if len(experiment_dirs) == 1:
        return experiment_dirs[0] / "posthoc" / "double_bootstrap" / "results.json"
    return (
        experiment_dirs[0].parent
        / "posthoc_aggregate"
        / "double_bootstrap"
        / "results.json"
    )


def _task_sample_flags(exp_dir: Path) -> dict[str, bool]:
    task_path = exp_dir / "task.json"
    if not task_path.exists():
        return {"exclude_financials": True, "exclude_microcap": True}
    task = json.loads(task_path.read_text())
    metadata = task.get("metadata", {})
    return {
        "exclude_financials": bool(metadata.get("exclude_financials", True)),
        "exclude_microcap": bool(metadata.get("exclude_microcap", True)),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_progress(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"created_at": _utc_now(), **payload}
    from siglab.utils.json import strict_json_dumps

    with path.open("a") as handle:
        handle.write(strict_json_dumps(row) + "\n")


def signal_spec_hash(signal_specs: list[dict]) -> str:
    payload = json.dumps(
        signal_specs,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _label_hash(labels) -> str:
    payload = json.dumps([str(value) for value in labels], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _date_string(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def _sign_for_spec(spec: dict) -> float:
    return -1.0 if spec.get("expected_sign", "positive") == "negative" else 1.0


def build_cache_metadata(
    *,
    signal_specs: list[dict],
    nw_lags: int,
    min_finite: int,
    start_date,
    sample_index,
    columns,
    matrix_shape: tuple[int, int] | list[int] | None = None,
    n_signals_kept: int | None = None,
    spec_hash: str | None = None,
) -> dict[str, object]:
    dates = _date_strings(pd.Index(sample_index))
    labels = list(columns)
    metadata: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "spec_hash": spec_hash or signal_spec_hash(signal_specs),
        "n_signals_requested": len(signal_specs),
        "nw_lags": int(nw_lags),
        "min_finite": int(min_finite),
        "start_date": _date_string(start_date),
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "date_count": len(dates),
        "dates_hash": _label_hash(dates),
        "columns_count": len(labels),
        "columns_hash": _label_hash(labels),
    }
    if matrix_shape is not None:
        metadata["matrix_shape"] = [int(value) for value in matrix_shape]
    if n_signals_kept is not None:
        metadata["n_signals_kept"] = int(n_signals_kept)
    return metadata


def _cache_build_metadata(metadata: dict) -> dict[str, object]:
    excluded = {"dates", "signal_specs", "signs", "observed_t"}
    return {key: value for key, value in metadata.items() if key not in excluded}


def _validate_cache_metadata(
    metadata: dict,
    *,
    expected_spec_hash: str,
    expected_metadata: dict[str, object] | None,
    matrix_shape: tuple[int, int],
) -> None:
    spec_hash = metadata.get("spec_hash")
    if spec_hash != expected_spec_hash:
        raise ValueError(
            f"beta cache hash mismatch: expected {expected_spec_hash}, got {spec_hash}"
        )

    stored_shape = metadata.get("matrix_shape")
    if stored_shape is not None and list(stored_shape) != [int(v) for v in matrix_shape]:
        raise ValueError(
            "beta cache metadata matrix_shape mismatch: "
            f"metadata has {stored_shape}, parquet has {list(matrix_shape)}"
        )

    if expected_metadata is None:
        return

    for key, expected_value in expected_metadata.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"beta cache metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


def write_beta_cache(cache_dir: str | Path, dataset: BetaMatrixDataset) -> None:
    from siglab.utils.json import write_strict_json

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    columns = [f"signal_{idx}" for idx in range(dataset.matrix.shape[1])]
    frame = pd.DataFrame(dataset.matrix, index=dataset.dates, columns=columns)
    frame.to_parquet(cache_path / "beta_matrix.parquet")
    metadata = {
        **dataset.metadata,
        "schema_version": dataset.metadata.get("schema_version", CACHE_SCHEMA_VERSION),
        "spec_hash": dataset.spec_hash,
        "matrix_shape": [int(value) for value in dataset.matrix.shape],
        "dates": dataset.dates,
        "signal_specs": dataset.signal_specs,
        "signs": dataset.signs.tolist(),
        "observed_t": dataset.observed_t.tolist(),
    }
    write_strict_json(cache_path / "metadata.json", metadata, indent=2)


def read_beta_cache(
    cache_dir: str | Path,
    expected_spec_hash: str,
    *,
    expected_metadata: dict[str, object] | None = None,
) -> BetaMatrixDataset:
    cache_path = Path(cache_dir)
    metadata = json.loads((cache_path / "metadata.json").read_text())
    frame = pd.read_parquet(cache_path / "beta_matrix.parquet")
    cached_dates = [str(value) for value in metadata.get("dates", [])]
    frame_dates = _date_strings(pd.Index(frame.index))
    if cached_dates and cached_dates != frame_dates:
        raise ValueError("beta cache dates do not match parquet index")
    if metadata.get("dates_hash") is not None:
        date_source = cached_dates if cached_dates else frame_dates
        actual_dates_hash = _label_hash(date_source)
        if metadata["dates_hash"] != actual_dates_hash:
            raise ValueError(
                "beta cache metadata dates_hash mismatch: "
                f"metadata has {metadata['dates_hash']}, dates hash to {actual_dates_hash}"
            )
    _validate_cache_metadata(
        metadata,
        expected_spec_hash=expected_spec_hash,
        expected_metadata=expected_metadata,
        matrix_shape=frame.shape,
    )
    spec_hash = metadata.get("spec_hash")
    return BetaMatrixDataset(
        matrix=frame.to_numpy(dtype=float),
        dates=[str(value) for value in metadata["dates"]],
        signal_specs=list(metadata["signal_specs"]),
        signs=np.asarray(metadata["signs"], dtype=float),
        observed_t=np.asarray(metadata["observed_t"], dtype=float),
        spec_hash=str(spec_hash),
        metadata=_cache_build_metadata(metadata),
    )


def load_signal_specs(
    experiment_dirs: list[str | Path],
    *,
    signal_scope: str = "evaluated",
) -> list[dict]:
    """Load bootstrap-eligible evaluated signals from experiment archives."""
    if signal_scope not in {"evaluated", "first_pass"}:
        raise ValueError("signal_scope must be 'evaluated' or 'first_pass'")
    flags_by_dir = {
        str(Path(path).resolve()): _task_sample_flags(Path(path))
        for path in experiment_dirs
    }
    source_rows = (
        collect_first_pass_discoveries(experiment_dirs)
        if signal_scope == "first_pass"
        else collect_evaluated_signals(experiment_dirs)
    )
    specs = []
    seen: set[tuple[str, str, str, str]] = set()
    for signal in source_rows:
        if signal.get("error") is not None:
            continue
        if signal.get("fmb_tstat") is None:
            continue
        expression = signal.get("expression")
        name = signal.get("name")
        if not expression or not name:
            continue
        source_dir = signal["source_experiment_dir"]
        expression_key = (
            signal.get("normalized_expression")
            or str(expression).strip().lower()
        )
        key = (
            str(signal.get("source_experiment", "")),
            str(source_dir),
            str(name),
            str(expression_key),
        )
        if key in seen:
            continue
        seen.add(key)
        specs.append({
            "name": str(name),
            "expression": str(expression),
            "expected_sign": str(signal.get("expected_sign", "positive")),
            "source_experiment_dir": source_dir,
            **flags_by_dir.get(
                source_dir,
                {"exclude_financials": True, "exclude_microcap": True},
            ),
        })
    return specs


def _date_strings(index: pd.Index) -> list[str]:
    out = []
    for value in index:
        if hasattr(value, "date"):
            out.append(str(value.date()))
        else:
            out.append(str(value))
    return out


def _monthly_fmb_betas(signal: pd.DataFrame, ret: pd.DataFrame) -> np.ndarray:
    signal_aligned, ret_aligned = signal.align(ret, join="inner", axis=None)
    x_values = signal_aligned.to_numpy(dtype=float)
    y_values = ret_aligned.to_numpy(dtype=float)
    betas = np.full(max(len(ret_aligned.index) - 1, 0), np.nan, dtype=float)

    for t in range(1, len(ret_aligned.index)):
        x = x_values[t - 1]
        y = y_values[t]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() <= 2:
            continue
        x_valid = x[valid]
        x_dm = x_valid - x_valid.mean()
        denom = float(np.dot(x_dm, x_dm))
        if not np.isfinite(denom) or denom <= 1e-15:
            continue
        betas[t - 1] = float(np.dot(x_dm, y[valid]) / denom)
    return betas


def _validate_beta_worker_count(workers: int) -> int:
    workers = int(workers)
    if not 1 <= workers <= MAX_BETA_WORKERS:
        raise ValueError(
            f"beta workers must be between 1 and {MAX_BETA_WORKERS}"
        )
    return workers


def _effective_beta_workers(workers: int, n_tasks: int) -> int:
    workers = _validate_beta_worker_count(workers)
    return min(workers, max(int(n_tasks), 1))


def _build_beta_sample(agent):
    from siglab.assay.sample import build_monthly_sample

    return build_monthly_sample(
        agent.ret_panel,
        agent.me_panel,
        agent.nyse_panel,
        start_date=agent.start_date,
    )


def _beta_context_from_sample(
    *,
    agent,
    sample,
    signal_specs: list[dict],
    nw_lags: int,
) -> BetaBuildContext:
    min_finite = max(3, nw_lags + 2)
    sample_index = sample.returns.index[1:]
    return BetaBuildContext(
        dates=_date_strings(sample_index),
        metadata=build_cache_metadata(
            signal_specs=signal_specs,
            nw_lags=nw_lags,
            min_finite=min_finite,
            start_date=agent.start_date,
            sample_index=sample_index,
            columns=sample.returns.columns,
        ),
        sample=sample,
    )


def beta_build_context_for_agent(
    *,
    agent,
    signal_specs: list[dict],
    nw_lags: int,
) -> BetaBuildContext:
    """Capture the small parent-side metadata needed after workers start."""
    sample = _build_beta_sample(agent)
    return _beta_context_from_sample(
        agent=agent,
        sample=sample,
        signal_specs=signal_specs,
        nw_lags=nw_lags,
    )


def _init_beta_worker() -> None:
    """Validate inherited data and prevent nested native-thread fan-out."""
    global _THREADPOOL_LIMITER
    if _BETA_WORKER_STATE is None:
        raise RuntimeError("beta worker started without inherited data")
    from threadpoolctl import threadpool_limits

    _THREADPOOL_LIMITER = threadpool_limits(limits=1)


def _build_beta_signal(agent, sample, task: tuple[int, dict, int]) -> dict:
    from siglab.assay.sample import prepare_signal_panel

    idx, spec, min_finite = task
    try:
        raw = agent.engine.execute(spec["expression"])
        prepared = prepare_signal_panel(
            raw,
            sample,
            financials=agent.fin_panel,
            exclude_financials=bool(spec.get("exclude_financials", True)),
            exclude_microcap=bool(spec.get("exclude_microcap", True)),
        )
        beta_values = _monthly_fmb_betas(
            prepared.signal,
            prepared.returns,
        )
        expected_months = max(len(sample.returns.index) - 1, 0)
        if len(beta_values) != expected_months:
            raise ValueError(
                "signal beta series length does not match the monthly sample"
            )
        finite_count = int(np.sum(np.isfinite(beta_values)))
        if finite_count <= min_finite:
            return {
                "index": idx,
                "status": "skipped",
                "reason": "insufficient_finite_betas",
                "finite_months": finite_count,
            }
        sign = _sign_for_spec(spec)
        return {
            "index": idx,
            "status": "complete",
            "finite_months": finite_count,
            "sign": sign,
            "betas": beta_values * sign,
        }
    except Exception as exc:
        return {
            "index": idx,
            "status": "skipped",
            "reason": str(exc),
        }


def _run_beta_worker(task: tuple[int, dict, int]) -> dict:
    if _BETA_WORKER_STATE is None:
        raise RuntimeError("beta worker was not initialized")
    return _build_beta_signal(
        _BETA_WORKER_STATE["agent"],
        _BETA_WORKER_STATE["sample"],
        task,
    )


def build_beta_matrix(
    *,
    agent,
    signal_specs: list[dict],
    progress_path: Path | None,
    nw_lags: int,
    workers: int = 1,
    context: BetaBuildContext | None = None,
) -> BetaMatrixDataset:
    """Execute signals and build the compact monthly signed-FMB beta matrix."""
    from siglab.assay.double_bootstrap import _mean_t_stats

    global _BETA_WORKER_STATE
    workers = _validate_beta_worker_count(workers)
    spec_hash = signal_spec_hash(signal_specs)
    min_finite = max(3, nw_lags + 2)
    sample = context.sample if context is not None else None
    if context is None:
        sample = _build_beta_sample(agent)
        context = _beta_context_from_sample(
            agent=agent,
            sample=sample,
            signal_specs=signal_specs,
            nw_lags=nw_lags,
        )
    if context.metadata.get("spec_hash") != spec_hash:
        raise ValueError("beta build context signal-spec hash mismatch")

    _write_progress(
        progress_path,
        {
            "event": "beta_build_start",
            "n_signals_requested": len(signal_specs),
            "spec_hash": spec_hash,
        },
    )

    oriented_betas: list[pd.Series] = []
    kept_specs: list[dict] = []
    signs: list[float] = []

    tasks = [
        (idx, dict(spec), min_finite)
        for idx, spec in enumerate(signal_specs)
    ]
    effective_workers = _effective_beta_workers(workers, len(tasks))
    if effective_workers == 1:
        if sample is None:
            sample = _build_beta_sample(agent)
        results = []
        for task in tasks:
            idx, spec, _ = task
            _write_progress(
                progress_path,
                {
                    "event": "signal_start",
                    "index": idx,
                    "name": spec.get("name"),
                },
            )
            results.append(_build_beta_signal(agent, sample, task))
    else:
        for idx, spec, _ in tasks:
            _write_progress(
                progress_path,
                {
                    "event": "signal_start",
                    "index": idx,
                    "name": spec.get("name"),
                },
            )
        if sample is None:
            sample = _build_beta_sample(agent)
        _BETA_WORKER_STATE = {"agent": agent, "sample": sample}
        try:
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=mp.get_context("fork"),
                initializer=_init_beta_worker,
            ) as pool:
                results = list(pool.map(_run_beta_worker, tasks))
        finally:
            _BETA_WORKER_STATE = None

    for result in results:
        idx = int(result["index"])
        spec = signal_specs[idx]
        if result["status"] == "complete":
            oriented_betas.append(
                pd.Series(result["betas"], index=context.dates)
            )
            signs.append(float(result["sign"]))
            kept_specs.append(dict(spec))
            _write_progress(
                progress_path,
                {
                    "event": "signal_complete",
                    "index": idx,
                    "name": spec.get("name"),
                    "finite_months": int(result["finite_months"]),
                },
            )
            continue

        event = {
            "event": "signal_skipped",
            "index": idx,
            "name": spec.get("name"),
            "reason": result["reason"],
        }
        if "finite_months" in result:
            event["finite_months"] = int(result["finite_months"])
        _write_progress(
            progress_path,
            event,
        )
        if result["reason"] != "insufficient_finite_betas":
            print(f"Skip {spec['name']}: {result['reason']}")

    if not oriented_betas:
        reason = "No signal beta series passed the finite-month threshold"
        _write_progress(
            progress_path,
            {
                "event": "beta_build_failed",
                "n_signals_kept": 0,
                "n_signals_requested": len(signal_specs),
                "reason": reason,
            },
        )
        raise ValueError(reason)

    beta_frame = pd.concat(oriented_betas, axis=1)
    matrix = beta_frame.to_numpy(dtype=float)
    metadata = {
        **context.metadata,
        "matrix_shape": [int(value) for value in matrix.shape],
        "n_signals_kept": len(kept_specs),
    }
    _write_progress(
        progress_path,
        {
            "event": "beta_build_complete",
            "n_signals_kept": len(kept_specs),
            "n_signals_requested": len(signal_specs),
        },
    )
    return BetaMatrixDataset(
        matrix=matrix,
        dates=list(context.dates),
        signal_specs=kept_specs,
        signs=np.asarray(signs, dtype=float),
        observed_t=_mean_t_stats(matrix, nw_lags=nw_lags),
        spec_hash=spec_hash,
        metadata=metadata,
    )


def estimate_dense_panel_gib(
    *,
    n_months: int,
    n_signals: int,
    n_stocks: int,
    bytes_per_value: int = 8,
) -> float:
    return (n_months * n_signals * n_stocks * bytes_per_value) / (1024 ** 3)


def build_bootstrap_arrays(
    *,
    agent,
    signal_specs: list[dict],
    exclude_microcap: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    """Execute signal expressions and return arrays for double-bootstrap."""
    from siglab.assay.sample import build_monthly_sample, prepare_signal_panel

    sample = build_monthly_sample(
        agent.ret_panel,
        agent.me_panel,
        agent.nyse_panel,
        start_date=agent.start_date,
    )

    panels: list[np.ndarray] = []
    kept_specs: list[dict] = []
    for spec in signal_specs:
        try:
            raw = agent.engine.execute(spec["expression"])
            prepared = prepare_signal_panel(
                raw,
                sample,
                financials=agent.fin_panel,
                exclude_financials=bool(spec.get("exclude_financials", True)),
                exclude_microcap=bool(spec.get("exclude_microcap", exclude_microcap)),
            )
            panels.append(prepared.signal.to_numpy(dtype=float))
            kept_specs.append(spec)
        except Exception as exc:
            print(f"Skip {spec['name']}: {exc}")

    return panels, sample.returns.to_numpy(dtype=float), kept_specs


def expected_cache_metadata_for_agent(
    *,
    agent,
    signal_specs: list[dict],
    nw_lags: int,
) -> dict[str, object]:
    return beta_build_context_for_agent(
        agent=agent,
        signal_specs=signal_specs,
        nw_lags=nw_lags,
    ).metadata


def beta_matrix_output_metadata(dataset: BetaMatrixDataset) -> dict[str, object]:
    metadata = dataset.metadata
    payload: dict[str, object] = {
        "n_months": int(dataset.matrix.shape[0]),
        "n_signals": int(dataset.matrix.shape[1]),
        "shape": [int(value) for value in dataset.matrix.shape],
        "spec_hash": dataset.spec_hash,
    }

    metadata_date_start = metadata.get("date_start") if metadata else None
    metadata_date_end = metadata.get("date_end") if metadata else None
    payload["date_start"] = (dataset.dates[0] if dataset.dates else None) or metadata_date_start
    payload["date_end"] = (dataset.dates[-1] if dataset.dates else None) or metadata_date_end

    for source_key, output_key in [
        ("n_signals_requested", "n_signals_requested"),
        ("n_signals_kept", "n_signals_kept"),
        ("nw_lags", "nw_lags"),
        ("min_finite", "min_finite"),
        ("schema_version", "cache_schema_version"),
    ]:
        if source_key in metadata:
            payload[output_key] = metadata[source_key]
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Harvey-Liu double-bootstrap from experiment archives",
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        help="Completed experiment folder. Can be provided multiple times.",
    )
    parser.add_argument("--experiment-dirs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--p0", type=float, default=0.10)
    parser.add_argument("--I", type=int, default=100)
    parser.add_argument("--J", type=int, default=200)
    parser.add_argument("--n-workers", type=int, default=12)
    parser.add_argument(
        "--beta-workers",
        type=int,
        choices=range(1, MAX_BETA_WORKERS + 1),
        default=DEFAULT_BETA_WORKERS,
        help=(
            "Processes used only to construct the compact beta matrix "
            f"(default {DEFAULT_BETA_WORKERS}, maximum {MAX_BETA_WORKERS})."
        ),
    )
    parser.add_argument("--max-signals", type=int)
    parser.add_argument(
        "--signal-scope",
        choices=["evaluated", "first_pass"],
        default="evaluated",
        help=(
            "Signal universe to load. Default 'evaluated' uses the full evaluated "
            "universe for search-penalty calibration; 'first_pass' is only for "
            "diagnostics."
        ),
    )
    parser.add_argument("--beta-cache-dir")
    parser.add_argument("--resume-beta-cache", action="store_true")
    parser.add_argument("--two-sided", action="store_true")
    parser.add_argument(
        "--nw-lags",
        type=int,
        default=None,  # sentinel: None = "not passed" (effective default 6)
        help=(
            f"Newey-West lags for the compact path (default {NW_LAGS_DEFAULT}). "
            "Not supported with --dense-panel-legacy, which hardcodes nw_lags=0."
        ),
    )
    parser.add_argument(
        "--dense-panel-legacy",
        action="store_true",
        help=(
            "Use the legacy dense-panel path instead of the compact "
            "beta-matrix path. Forces two-sided ranking/testing with "
            "nw_lags=0 and has no parameters to change them: rejects an "
            "explicit --nw-lags and requires --two-sided (see "
            "check_legacy_flag_compat)."
        ),
    )
    parser.add_argument(
        "--t-grid",
        nargs="+",
        type=float,
        default=None,
        help="Optional t-stat threshold grid.",
    )
    return parser


def check_legacy_flag_compat(args: argparse.Namespace) -> str | None:
    """Reject flag combinations the legacy dense-panel path silently ignored.

    The legacy function (``run_double_bootstrap``) hardcodes
    ranking_mode='two_sided', testing_mode='two_sided', nw_lags=0 and has
    no parameters to change them; before this guard, --nw-lags and signed
    mode were silently dropped, making legacy runs incomparable with
    compact runs launched with identical flags (P2-16).
    """
    if not args.dense_panel_legacy:
        return None
    problems = []
    if args.nw_lags is not None:
        problems.append(
            f"--nw-lags {args.nw_lags} was passed but the legacy path "
            "hardcodes nw_lags=0"
        )
    if not args.two_sided:
        problems.append(
            "signed mode (the default without --two-sided) is not available; "
            "the legacy path hardcodes two-sided ranking and testing"
        )
    if not problems:
        return None
    return (
        "--dense-panel-legacy only supports two-sided mode with nw_lags=0: "
        + "; ".join(problems)
        + ". Pass --two-sided and omit --nw-lags, or drop "
        "--dense-panel-legacy to use the compact beta-matrix path."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    experiment_dirs = _experiment_dirs(args)
    if not experiment_dirs:
        parser.error("provide --experiment-dir or --experiment-dirs")
    try:
        experiment_dirs = validate_experiment_dirs(experiment_dirs)
    except ValueError as exc:
        parser.error(str(exc))

    legacy_error = check_legacy_flag_compat(args)
    if legacy_error:
        parser.error(legacy_error)
    nw_lags = args.nw_lags if args.nw_lags is not None else NW_LAGS_DEFAULT

    load_dotenv(REPO_ROOT / ".env")
    from siglab.assay.double_bootstrap import (
        run_double_bootstrap,
        run_double_bootstrap_from_matrix,
    )
    from siglab.lab.runner import build_agent_from_env

    out = Path(args.output) if args.output else _default_output_path(experiment_dirs)
    cache_dir = Path(args.beta_cache_dir) if args.beta_cache_dir else out.parent / "beta_cache"
    progress_path = out.parent / "progress.jsonl"

    signal_specs = load_signal_specs(
        experiment_dirs,
        signal_scope=args.signal_scope,
    )
    if args.max_signals is not None:
        signal_specs = signal_specs[:args.max_signals]
    if not signal_specs:
        raise SystemExit(f"No {args.signal_scope} signals found in experiment archives")

    print(f"Loaded {len(signal_specs)} {args.signal_scope} signals")
    agent = build_agent_from_env()
    n_months = int(np.sum(agent.ret_panel.index >= agent.start_date))
    dense_gib = estimate_dense_panel_gib(
        n_months=n_months,
        n_signals=len(signal_specs),
        n_stocks=len(agent.ret_panel.columns),
    )
    print(f"Dense legacy panel estimate: {dense_gib:.1f} GiB")
    if args.dense_panel_legacy and dense_gib > 4.0:
        raise SystemExit(
            "Dense legacy panel estimate exceeds 4 GiB; remove "
            "--dense-panel-legacy to use the compact beta-matrix path."
        )

    dataset = None
    if args.dense_panel_legacy:
        print(
            "Legacy dense-panel path: effective settings "
            "ranking_mode=two_sided testing_mode=two_sided nw_lags=0"
        )
        panels, ret_panel, kept_specs = build_bootstrap_arrays(
            agent=agent,
            signal_specs=signal_specs,
        )
        if not panels:
            raise SystemExit("No executable signal panels")
        result = run_double_bootstrap(
            panels,
            ret_panel,
            p0=args.p0,
            I=args.I,
            J=args.J,
            t_grid=args.t_grid,
            n_workers=args.n_workers,
        )
    else:
        expected_hash = signal_spec_hash(signal_specs)
        beta_context = beta_build_context_for_agent(
            agent=agent,
            signal_specs=signal_specs,
            nw_lags=nw_lags,
        )
        expected_metadata = beta_context.metadata
        if args.resume_beta_cache:
            try:
                dataset = read_beta_cache(
                    cache_dir,
                    expected_hash,
                    expected_metadata=expected_metadata,
                )
                print(f"Loaded beta cache -> {cache_dir}")
            except (FileNotFoundError, ValueError) as exc:
                print(f"Beta cache unavailable: {exc}; rebuilding")
        if dataset is None:
            effective_beta_workers = _effective_beta_workers(
                args.beta_workers,
                len(signal_specs),
            )
            dataset = build_beta_matrix(
                agent=agent,
                signal_specs=signal_specs,
                progress_path=progress_path,
                nw_lags=nw_lags,
                workers=effective_beta_workers,
                context=beta_context,
            )
            write_beta_cache(cache_dir, dataset)
            print(f"Wrote beta cache -> {cache_dir}")
        beta_context = None
        agent = None
        gc.collect()
        kept_specs = dataset.signal_specs
        mode = "two_sided" if args.two_sided else "signed"
        result = run_double_bootstrap_from_matrix(
            dataset.matrix,
            p0=args.p0,
            I=args.I,
            J=args.J,
            t_grid=args.t_grid,
            n_workers=args.n_workers,
            ranking_mode=mode,
            testing_mode=mode,
            nw_lags=nw_lags,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "3",
        "semantics": POSTHOC_RESULTS_SEMANTICS,
        "experiment_dirs": [str(path) for path in experiment_dirs],
        "signal_scope": args.signal_scope,
        "signals": kept_specs,
        "double_bootstrap": asdict(result),
        "method": {
            "paper": "Harvey and Liu, False (and Missed) Discoveries in Financial Economics",
            "matrix": "monthly_signed_fmb_beta",
            "beta_cache_dir": str(cache_dir),
            "progress_path": str(progress_path),
            "dense_panel_legacy": bool(args.dense_panel_legacy),
        },
    }
    if dataset is not None:
        payload["beta_matrix"] = beta_matrix_output_metadata(dataset)
    from siglab.utils.json import write_strict_json

    write_strict_json(out, payload, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
