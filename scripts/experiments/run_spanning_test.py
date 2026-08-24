"""Spanning tests against Chen-Zimmermann anomalies.

For each correct-sign significant discovery, tests incremental predictive power after
controlling for the most correlated existing anomalies from the full
CZ dataset (209 predictors, no cherry-picking).

CZ panels are loaded ONCE in the main process (~33GB) and shared across
ThreadPoolExecutor workers (threads share memory, no duplication).

Usage
-----
    python scripts/experiments/run_spanning_test.py \
        --experiment-dir data/experiments/<timestamp>_<run>_<pair>
    python scripts/experiments/run_spanning_test.py \
        --experiment-dirs data/experiments/<exp-a> data/experiments/<exp-b>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from siglab.assay.sample import POSTHOC_RESULTS_SEMANTICS

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from tqdm import tqdm

CACHE_DIR  = REPO_ROOT / "data" / "cache"
N_THREADS  = 8  # threads share memory; numpy releases GIL

# Rank-aware greedy control selection changes control sets and conditional
# t-statistics without changing the shared sample semantics used by the other
# posthoc suites. Keep a spanning-specific suffix so --resume refuses to mix
# pre-fix and post-fix rows while unrelated suite artifacts remain compatible.
RESULTS_SEMANTICS = f"{POSTHOC_RESULTS_SEMANTICS}+cz-rank-aware-v1"


def _experiment_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = []
    if args.experiment_dir:
        dirs.extend(args.experiment_dir)
    if args.experiment_dirs:
        dirs.extend(args.experiment_dirs)
    return [Path(path) for path in dirs]


def _default_output_path(experiment_dirs: list[Path]) -> Path:
    if len(experiment_dirs) == 1:
        return experiment_dirs[0] / "posthoc" / "spanning" / "results.json"
    return experiment_dirs[0].parent / "posthoc_aggregate" / "spanning" / "results.json"


def _read_json_list(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    try:
        rows = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return rows if isinstance(rows, list) else []


def _completed_signal_ids(output_path: Path) -> set[str]:
    # Errored rows do not count as completed so --resume re-runs them;
    # NaN-without-error rows (insufficient joint coverage) stay completed.
    return {
        str(row["signal_id"])
        for row in _read_json_list(output_path)
        if isinstance(row, dict) and row.get("signal_id") and not row.get("error")
    }


def _resume_semantics_error(output_path: Path) -> str | None:
    """Refusal message if `output_path` predates RESULTS_SEMANTICS, else None.

    --resume accretively merges prior spanning-test rows into new output
    with no semantics marker of its own, so resuming over a file written
    before this branch's value changes would silently mix old-semantics and
    new-semantics rows in one file. A missing/empty file has nothing to mix
    and is always fine to resume over.
    """
    if not output_path.exists():
        return None
    try:
        rows = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return (
            f"Cannot resume from {output_path}: the existing output is "
            f"unreadable or invalid JSON ({exc}). Move or delete this file, "
            "then rerun without --resume to start a fresh run."
        )
    if not isinstance(rows, list) or not rows:
        return None
    if all(
        isinstance(row, dict) and row.get("semantics") == RESULTS_SEMANTICS
        for row in rows
    ):
        return None
    return (
        f"{output_path} contains rows from an older spanning-test semantics "
        f"version (one or more rows are missing the {RESULTS_SEMANTICS!r} "
        "tag). Resuming would silently mix old-semantics and new-semantics "
        "rows in the same file. Choose a fresh output path and rerun without "
        "--resume to start a correctly-tagged run."
    )


def _stamp_new_rows(rows: list[dict]) -> list[dict]:
    """Tag freshly-computed rows with the current results semantics."""
    for row in rows:
        row["semantics"] = RESULTS_SEMANTICS
    return rows


def _retry_once_on_error(fn, *args, **kwargs):
    """Call fn once more if its result carries an error.

    Transient failures (thread races building lazy pandas index engines)
    succeed on the second attempt; deterministic errors fail twice and keep
    their error message.
    """
    result = fn(*args, **kwargs)
    if getattr(result, "error", None):
        result = fn(*args, **kwargs)
    return result


def _pending_survivors(survivors: list[dict], completed_ids: set[str]) -> list[dict]:
    return [
        sig for sig in survivors
        if str(sig.get("signal_id")) not in completed_ids
    ]


def _write_json_list(output_path: Path, rows: list[dict], *, default=None) -> None:
    from siglab.utils.json import write_strict_json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(output_path, rows, indent=2)


# ── Signal loading ───────────────────────────────────────────────────────

def load_survivors(experiment_dirs: list[Path]) -> list[dict]:
    from siglab.lab.posthoc import collect_horse_race_survivors

    survivors = collect_horse_race_survivors(experiment_dirs)
    print(f"Loaded {len(survivors)} horse-race survivors from experiment archives")
    return survivors


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CZ spanning tests")
    parser.add_argument("--experiment-dir", action="append")
    parser.add_argument("--experiment-dirs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=N_THREADS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--corr-sample-every", type=int, default=12)
    parser.add_argument("--min-valid-months", type=int, default=24)
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

    # ── Load survivors before heavy CRSP/CZ data ────────────────────────
    survivors = load_survivors(experiment_dirs)
    if not survivors:
        print("No horse-race survivors found.")
        return
    completed_ids = _completed_signal_ids(out) if args.resume else set()
    if completed_ids:
        print(f"Resuming spanning: skipping {len(completed_ids)} completed signals")
        survivors = _pending_survivors(survivors, completed_ids)
    if not survivors:
        previous_results = _read_json_list(out) if args.resume else []
        _write_json_list(out, previous_results)
        print(f"No pending spanning signals; preserved existing output -> {out}")
        return

    import pandas as pd
    from siglab.data import cache
    from siglab.data import crsp as crsp_mod
    from siglab.agent.executor import SignalEngine
    from siglab.assay.spanning import run_spanning_test
    from siglab.assay.sample import (
        build_monthly_sample,
        normalize_monthly_panel,
        prepare_signal_panel,
    )
    from siglab.utils.panel import warm_index_engines

    cache.set_cache_dir(CACHE_DIR)

    # ── Load data (single copy in main process) ─────────────────────────
    print("Loading CRSP panels...")
    crsp_panels = crsp_mod.get_crsp_panels(use_cache=True)
    sample = build_monthly_sample(
        crsp_panels["ret"], crsp_panels["me"], crsp_panels["nyse"]
    )

    print("Loading Compustat-CCM...")
    crsp_raw = cache.load_panel("crsp_1960_2024_raw")
    crsp_raw["date"]  = pd.to_datetime(crsp_raw["date"])
    crsp_raw["month"] = crsp_raw["date"].dt.month
    crsp_raw["year"]  = crsp_raw["date"].dt.year
    comp_ccm = cache.load_panel("comp_ccm_1960_2024")

    sic_panel, fin_panel = None, None
    if "siccd" in crsp_raw.columns:
        sic_panel = crsp_raw.pivot_table(
            values="siccd", index="date", columns="permno", aggfunc="first"
        )
        sic_panel.index = pd.to_datetime(sic_panel.index)
        fin_panel = (sic_panel >= 6000) & (sic_panel < 7000)

    engine = SignalEngine(comp=comp_ccm, crsp=crsp_raw, sic_panel=sic_panel)

    # ── Load CZ anomalies (single copy, shared by threads) ──────────────
    print("Loading CZ predictors (209 anomalies)...")
    from siglab.data.cz_anomalies import cz_to_all_panels
    cz_panels = {
        name: normalize_monthly_panel(panel, name=f"CZ {name}")
        for name, panel in cz_to_all_panels().items()
    }
    print(f"  {len(cz_panels)} CZ panels loaded into memory")

    # ── Pre-compute signal panels ────────────────────────────────────────
    print("Executing survivor signal expressions...")
    sig_panels: dict[str, pd.DataFrame] = {}
    for sig in tqdm(survivors, desc="Signals", unit="sig"):
        name = sig["name"]
        sid = sig["signal_id"]
        try:
            raw = engine.execute(sig["expression"])
            prepared = prepare_signal_panel(
                raw,
                sample,
                financials=fin_panel,
                exclude_financials=bool(sig.get("exclude_financials", True)),
                exclude_microcap=bool(sig.get("exclude_microcap", True)),
            )
            sig_panels[sid] = prepared.signal
        except Exception as e:
            tqdm.write(f"  Skip {name}: {e}")

    # ── Run spanning tests (ThreadPoolExecutor — shared memory) ──────────
    # Panels shared across workers need their index engines built before
    # the pool starts (lazy engine construction is not thread-safe).
    warm_index_engines(sample.returns, *cz_panels.values())

    print(f"\nRunning {len(sig_panels)} spanning tests ({args.workers} threads, top_k={args.top_k})...\n")

    def _run_one(sig: dict) -> dict:
        name = sig["name"]
        sid = sig["signal_id"]
        if sid not in sig_panels:
            return {
                "name": name,
                "signal_id": sid,
                "status": "error",
                "reason": "panel_build_error",
                "error": "no panel",
            }
        try:
            result = _retry_once_on_error(
                run_spanning_test,
                name, sig_panels[sid], sample.returns, cz_panels,
                top_k=args.top_k,
                corr_sample_every=args.corr_sample_every,
                expected_sign=sig.get("expected_sign", "positive"),
                min_valid_months=args.min_valid_months,
            )
            return {
                "name": name,
                "signal_id": sid,
                "expression": sig.get("expression", ""),
                "pair": sig.get("pair_name", ""),
                "fmb_t_original": sig.get("fmb_tstat"),
                "fmb_t_univariate": result.fmb_t_univariate,
                "fmb_t_conditional": result.fmb_t_conditional,
                "n_controls": result.n_controls,
                "top_control": result.control_names[0] if result.control_names else None,
                "top_corr": result.control_corrs[0] if result.control_corrs else None,
                "control_names": result.control_names,
                "control_corrs": result.control_corrs,
                "n_identified_months": result.n_identified_months,
                "n_valid_months": result.n_valid_months,
                "skipped_controls": result.skipped_controls,
                "survives_196": result.survives_196,
                "survives_300": result.survives_300,
                "absolute_survives_196": result.absolute_survives_196,
                "absolute_survives_300": result.absolute_survives_300,
                "direction_consistent": result.direction_consistent,
                "status": result.status,
                "reason": result.reason,
                "error": result.error,
            }
        except Exception as e:
            return {
                "name": name,
                "signal_id": sid,
                "expression": sig.get("expression", ""),
                "pair": sig.get("pair_name", ""),
                "status": "error",
                "reason": "spanning_runner_exception",
                "error": str(e),
            }

    all_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, sig): sig["signal_id"] for sig in survivors}
        with tqdm(total=len(futures), desc="Spanning", unit="sig") as pbar:
            for future in as_completed(futures):
                res = future.result()
                all_results.append(res)
                tc = res.get("fmb_t_conditional")
                top = res.get("top_control", "?") or "?"
                rho = res.get("top_corr", 0) or 0
                flag = " ***" if res.get("survives_196") else ""
                tc_s = f"{tc:5.2f}" if tc is not None else "  err"
                tqdm.write(f"  {res['name']:<35s} t_cond={tc_s}{flag}  closest={top}(r={rho:.2f})")
                pbar.update(1)

    # ── Save results ─────────────────────────────────────────────────────
    previous_results = _read_json_list(out) if args.resume else []
    all_results = previous_results + _stamp_new_rows(all_results)
    _write_json_list(out, all_results)
    print(f"\nSaved → {out}")

    # ── Summary ──────────────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("status") == "ok"]
    n_surv_196 = sum(1 for r in valid if r.get("survives_196"))
    n_surv_300 = sum(1 for r in valid if r.get("survives_300"))

    print(f"\n{'='*70}")
    print(f"SPANNING TEST SUMMARY (top_k={args.top_k})")
    print(f"{'='*70}")
    print(f"  Total survivors tested: {len(survivors)}")
    print(f"  Valid results: {len(valid)}")
    print(f"  Survive |t|>1.96 after CZ controls: {n_surv_196}/{len(valid)}")
    print(f"  Survive |t|>3.00 after CZ controls: {n_surv_300}/{len(valid)}")

    print(f"\n{'Signal':<35s} {'t_orig':>7s} {'t_cond':>7s} {'closest_CZ':>20s} {'rho':>6s}")
    print("-" * 80)
    for r in sorted(valid, key=lambda x: abs(x.get("fmb_t_conditional", 0)), reverse=True):
        flag = " ***" if r.get("survives_196") else ""
        tc = r.get("fmb_t_conditional", 0) or 0
        to = r.get("fmb_t_original", 0) or 0
        top = r.get("top_control", "?") or "?"
        rho = r.get("top_corr", 0) or 0
        print(f"  {r['name']:<33s} {to:7.2f} {tc:7.2f} {top:>20s} {rho:6.2f}{flag}")


if __name__ == "__main__":
    main()
