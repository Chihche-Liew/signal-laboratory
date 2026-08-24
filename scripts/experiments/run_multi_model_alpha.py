"""Multi-model factor alpha tests for correct-sign significant discoveries.

Tests each survivor's long-short portfolio against three factor models:
  5 = FF5  (Mkt-RF, SMB, HML, RMW, CMA)
  6 = FF6  (+ Mom)
  7 = q-factor (Mkt, ME, I/A, ROE)

Results are saved alongside existing robustness outputs.

Usage
-----
    python scripts/experiments/run_multi_model_alpha.py \
        --experiment-dir data/experiments/<timestamp>_<run>_<pair>
    python scripts/experiments/run_multi_model_alpha.py \
        --experiment-dirs data/experiments/<exp-a> data/experiments/<exp-b>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from siglab.assay.sample import POSTHOC_RESULTS_SEMANTICS

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from tqdm import tqdm

CACHE_DIR  = REPO_ROOT / "data" / "cache"
N_WORKERS  = 8

# Must match the start/end that load_data() passes to get_crsp_panels() —
# kept as explicit constants (rather than relying on get_crsp_panels'
# defaults) so the panel cache_key and the worker initargs derived from it
# can never diverge (see crsp.crsp_cache_key).
CRSP_START = "1960-01-01"
CRSP_END   = "2024-12-31"

MODELS = [5, 6, 7]
MODEL_NAMES = {5: "FF5", 6: "FF6", 7: "q4"}

# This branch changed the VALUES several posthoc scripts compute (grs_p,
# subsample/decay labels, wls_fmb_t). Every row this script writes carries
# this tag so --resume can detect and refuse to silently mix pre-fix rows
# with post-fix rows in the same results file.
RESULTS_SEMANTICS = POSTHOC_RESULTS_SEMANTICS


def _experiment_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = []
    if args.experiment_dir:
        dirs.extend(args.experiment_dir)
    if args.experiment_dirs:
        dirs.extend(args.experiment_dirs)
    return [Path(path) for path in dirs]


def _default_output_path(experiment_dirs: list[Path]) -> Path:
    if len(experiment_dirs) == 1:
        return experiment_dirs[0] / "posthoc" / "multi_model_alpha" / "results.json"
    return (
        experiment_dirs[0].parent
        / "posthoc_aggregate"
        / "multi_model_alpha"
        / "results.json"
    )


def _read_json_list(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    try:
        rows = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return rows if isinstance(rows, list) else []


def _resume_semantics_error(output_path: Path) -> str | None:
    """Refusal message if `output_path` predates RESULTS_SEMANTICS, else None.

    --resume accretively merges prior rows into new output with no
    semantics marker of its own, so resuming over a file written before
    this branch's value changes (grs_p, wls_fmb_t, ...) would silently mix
    old-semantics and new-semantics rows in one file. A missing/empty file
    has nothing to mix and is always fine to resume over.
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
        f"{output_path} predates the 2026-07 posthoc semantics fixes (one or "
        f"more rows are missing the {RESULTS_SEMANTICS!r} tag). Resuming "
        "would silently mix old-semantics and new-semantics rows (grs_p, "
        "wls_fmb_t, ...) in the same file. Move or delete this file, then "
        "rerun without --resume to start a fresh, correctly-tagged run."
    )


# Error-bearing keys a row may carry. Historical rows written before error
# persistence lack all of them and therefore still count as complete.
_ERROR_KEYS = tuple(f"{MODEL_NAMES[m]}_error" for m in MODELS) + (
    "wls_fmb_error",
    "error",
)


def _row_is_complete(row: dict) -> bool:
    """Complete = has a signal_id and carries no error anywhere.

    All-null metrics with no error are a legitimate degenerate outcome
    (run_univ_sort returns empty results for unsortable signals) and must
    NOT trigger a retry.
    """
    return bool(row.get("signal_id")) and not any(row.get(k) for k in _ERROR_KEYS)


def _completed_signal_ids(output_path: Path) -> set[str]:
    return {
        str(row["signal_id"])
        for row in _read_json_list(output_path)
        if isinstance(row, dict) and _row_is_complete(row)
    }


def _assemble_rows(survivors: list[dict], result_map: dict[str, dict]) -> list[dict]:
    """Flatten worker results into output rows, PRESERVING error fields."""
    rows = []
    for sig in survivors:
        signal_id = sig["signal_id"]
        row = {
            "name":         sig["name"],
            "signal_id":    signal_id,
            "expression":   sig.get("expression", ""),
            "pair":         sig.get("pair_name", ""),
            "full_fmb_t":   sig.get("fmb_tstat"),
            "horse_race_t": sig.get("horse_race_t"),
        }
        res = result_map.get(signal_id)
        if res is None:
            res = {"results": {}, "error": "worker returned no result"}
        row["error"] = res.get("error")
        row["status"] = res.get("status", "error" if res.get("error") else "ok")
        row["reason"] = res.get("reason")
        for m in MODELS:
            mn = MODEL_NAMES[m]
            model_res = res.get("results", {}).get(mn, {})
            row[f"{mn}_alpha"]  = model_res.get("alpha")
            row[f"{mn}_talpha"] = model_res.get("talpha")
            row[f"{mn}_sharpe"] = model_res.get("sharpe")
            row[f"{mn}_grs_p"]  = model_res.get("grs_pval")
            row[f"{mn}_error"]  = model_res.get("error")
            row[f"{mn}_status"] = model_res.get("status")
            row[f"{mn}_reason"] = model_res.get("reason")
            row[f"{mn}_direction_consistent"] = model_res.get("direction_consistent")
            row[f"{mn}_survives_196"] = model_res.get("survives_196")
        wls_res = res.get("results", {}).get("WLS_FMB", {})
        row["wls_fmb_t"] = wls_res.get("fmb_tstat")
        row["wls_fmb_error"] = wls_res.get("error")
        row["wls_fmb_status"] = wls_res.get("status")
        row["wls_fmb_reason"] = wls_res.get("reason")
        row["semantics"] = RESULTS_SEMANTICS
        rows.append(row)
    return rows


def _pending_survivors(survivors: list[dict], completed_ids: set[str]) -> list[dict]:
    return [
        sig for sig in survivors
        if str(sig.get("signal_id")) not in completed_ids
    ]


def _write_json_list(output_path: Path, rows: list[dict], *, default=None) -> None:
    from siglab.utils.json import write_strict_json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(output_path, rows, indent=2)


# ── Worker state ─────────────────────────────────────────────────────────

_WORKER_STATE = None


def _init_worker(
    repo_src: str, cache_dir: str,
    comp_ccm_key: str, crsp_raw_key: str,
    ret_key: str, me_key: str, ff_key: str, nyse_key: str,
    q_key: str, fin_key: str | None, sic_key: str | None,
) -> None:
    global _WORKER_STATE
    import warnings; warnings.filterwarnings("ignore")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    import pandas as pd
    from siglab.data import cache as cache_mod
    from siglab.agent.executor import SignalEngine
    from siglab.assay.sample import (
        build_monthly_sample,
        normalize_factor_panel,
    )

    cache_mod.set_cache_dir(cache_dir)
    comp_ccm  = cache_mod.load_panel(comp_ccm_key)
    crsp_raw  = cache_mod.load_panel(crsp_raw_key)
    crsp_raw["date"]  = pd.to_datetime(crsp_raw["date"])
    crsp_raw["month"] = crsp_raw["date"].dt.month
    crsp_raw["year"]  = crsp_raw["date"].dt.year
    ret_panel  = cache_mod.load_panel(ret_key)
    me_panel   = cache_mod.load_panel(me_key)
    ff_factors = cache_mod.load_panel(ff_key)
    nyse_panel = cache_mod.load_panel(nyse_key)
    q_factors  = cache_mod.load_panel(q_key)
    fin_panel  = cache_mod.load_panel(fin_key).astype(bool) if fin_key else None
    sic_panel  = cache_mod.load_panel(sic_key) if sic_key else None

    engine = SignalEngine(comp=comp_ccm, crsp=crsp_raw, sic_panel=sic_panel)
    sample = build_monthly_sample(ret_panel, me_panel, nyse_panel)
    _WORKER_STATE = {
        "engine": engine,
        "sample": sample,
        "ff_factors": normalize_factor_panel(
            ff_factors, name="ff_factors"
        ).reindex(sample.returns.index),
        "q_factors": normalize_factor_panel(
            q_factors, name="q_factors"
        ).reindex(sample.returns.index),
        "fin_panel": fin_panel,
    }


def _eval_multi_model_worker(task: tuple) -> dict:
    """Evaluate one signal across all factor models."""
    (
        signal_id,
        name,
        expression,
        expected_sign,
        exclude_financials,
        exclude_microcap,
    ) = task
    from siglab.portfolio.sorts import run_univ_sort
    from siglab.assay.sample import (
        direction_matches,
        finite_or_none,
        prepare_signal_panel,
    )

    try:
        raw = _WORKER_STATE["engine"].execute(expression)
        prepared = prepare_signal_panel(
            raw,
            _WORKER_STATE["sample"],
            financials=_WORKER_STATE["fin_panel"],
            exclude_financials=exclude_financials,
            exclude_microcap=exclude_microcap,
        )

        sig = prepared.signal
        ret = prepared.returns
        me = prepared.market_equity
        nyse = prepared.nyse
        ff  = _WORKER_STATE["ff_factors"]
        qf  = _WORKER_STATE["q_factors"]

        results: dict[str, dict] = {}
        for model_id in MODELS:
            try:
                sort_result = run_univ_sort(
                    ret, sig, me, ff,
                    n_ptf=5, weighting="value", factor_model=model_id,
                    nyse_indicator=nyse, add_long_short=True,
                    q_factors=qf if model_id == 7 else None,
                )
                ls_idx = -1  # LS portfolio is last column
                alpha = (
                    finite_or_none(sort_result.alpha[ls_idx] * 12)
                    if sort_result.alpha is not None else None
                )
                talpha = (
                    finite_or_none(sort_result.talpha[ls_idx])
                    if sort_result.talpha is not None else None
                )
                sharpe = (
                    finite_or_none(sort_result.sharpe[ls_idx])
                    if sort_result.sharpe is not None else None
                )
                grs_p = finite_or_none(sort_result.grs_pval)
                status = "ok" if talpha is not None else "not_estimable"
                sign_ok = direction_matches(talpha, expected_sign)

                results[MODEL_NAMES[model_id]] = {
                    "alpha": alpha,
                    "talpha": talpha,
                    "sharpe": sharpe,
                    "grs_pval": grs_p,
                    "status": status,
                    "reason": None if status == "ok" else "non_finite_alpha_t",
                    "direction_consistent": sign_ok,
                    "survives_196": bool(
                        talpha is not None and abs(talpha) > 1.96 and sign_ok
                    ),
                    "error": None,
                }
            except Exception as exc:
                results[MODEL_NAMES[model_id]] = {
                    "alpha": None, "talpha": None,
                    "sharpe": None, "grs_pval": None,
                    "status": "error",
                    "reason": "factor_model_exception",
                    "direction_consistent": False,
                    "survives_196": False,
                    "error": str(exc),
                }

        # WLS Fama-MacBeth (Part 0.2: ME-weighted cross-sectional regression)
        try:
            from siglab.factor_model.fama_macbeth import run_fama_macbeth
            wls_fmb = run_fama_macbeth({name: sig}, ret, n_lags=1, weights=me)
            wls_t = (
                finite_or_none(wls_fmb.tstat[1])
                if wls_fmb is not None and len(wls_fmb.tstat) > 1
                else None
            )
            results["WLS_FMB"] = {
                "fmb_tstat": wls_t,
                "status": "ok" if wls_t is not None else "not_estimable",
                "reason": None if wls_t is not None else "non_finite_wls_fmb_t",
                "error": None,
            }
        except Exception as exc:
            results["WLS_FMB"] = {
                "fmb_tstat": None,
                "status": "error",
                "reason": "wls_fmb_exception",
                "error": str(exc),
            }

        statuses = [result.get("status") for result in results.values()]
        overall_status = "error" if "error" in statuses else (
            "not_estimable" if "not_estimable" in statuses else "ok"
        )
        return {
            "signal_id": signal_id,
            "name": name,
            "results": results,
            "status": overall_status,
            "reason": None if overall_status == "ok" else "one_or_more_models_incomplete",
            "error": None,
        }

    except Exception as exc:
        return {
            "signal_id": signal_id,
            "name": name,
            "results": {},
            "status": "error",
            "reason": "signal_evaluation_exception",
            "error": str(exc),
        }


# ── Data loading ─────────────────────────────────────────────────────────

def load_data() -> dict:
    from siglab.data import cache, crsp as crsp_mod
    from siglab.data.factors import download_ff_factors, download_q_factors
    import pandas as pd

    cache.set_cache_dir(CACHE_DIR)

    crsp_panels = crsp_mod.get_crsp_panels(CRSP_START, CRSP_END, use_cache=True)
    ret_panel  = crsp_panels["ret"]
    me_panel   = crsp_panels["me"]
    nyse_panel = crsp_panels["nyse"]

    crsp_raw = cache.load_panel("crsp_1960_2024_raw")
    crsp_raw["date"]  = pd.to_datetime(crsp_raw["date"])
    crsp_raw["month"] = crsp_raw["date"].dt.month
    crsp_raw["year"]  = crsp_raw["date"].dt.year

    ff_factors = download_ff_factors(use_cache=True)
    q_factors  = download_q_factors(use_cache=True)
    comp = cache.load_panel("compustat_annual_1960_2024")
    ccm  = cache.load_panel("ccm_link_1960_2024")

    comp_ccm = comp.merge(
        ccm[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner"
    )
    comp_ccm = comp_ccm[
        (comp_ccm["datadate"] >= comp_ccm["linkdt"])
        & (comp_ccm["datadate"] <= comp_ccm["linkenddt"])
    ]
    comp_ccm = comp_ccm.drop(columns=["linkdt", "linkenddt"])
    comp_ccm["year"] = comp_ccm["datadate"].dt.year
    comp_ccm = comp_ccm.sort_values(["permno", "year", "datadate"])
    comp_ccm = comp_ccm.drop_duplicates(subset=["permno", "year"], keep="last")
    comp_ccm["form_year"] = comp_ccm["year"] + 1

    sic_panel = None
    fin_panel = None
    if "siccd" in crsp_raw.columns:
        sic_panel = crsp_raw.pivot_table(
            values="siccd", index="date", columns="permno", aggfunc="first"
        )
        sic_panel.index = pd.to_datetime(sic_panel.index)
        fin_panel = (sic_panel >= 6000) & (sic_panel < 7000)

    # Ensure derived panels are cached for workers
    comp_ccm_key = "comp_ccm_1960_2024"
    if not cache.panel_exists(comp_ccm_key):
        cache.save_panel(comp_ccm_key, comp_ccm)

    q_key = "q_factors"
    if not cache.panel_exists(q_key):
        cache.save_panel(q_key, q_factors)

    fin_panel_key = None
    if fin_panel is not None:
        fin_panel_key = "fin_panel_1960_2024"
        if not cache.panel_exists(fin_panel_key):
            cache.save_panel(fin_panel_key, fin_panel.astype(float))

    sic_panel_key = None
    if sic_panel is not None:
        sic_panel_key = "sic_panel_1960_2024"
        if not cache.panel_exists(sic_panel_key):
            cache.save_panel(sic_panel_key, sic_panel)

    # Derive from the same source of truth get_crsp_panels() uses above, so
    # workers can never silently reload a stale/orphaned v1 bundle (P2-11
    # follow-up: the v1->v2 cache bump previously left these as hardcoded
    # v1 literals).
    crsp_key = crsp_mod.crsp_cache_key(CRSP_START, CRSP_END)
    worker_initargs = (
        str(REPO_ROOT / "src"),
        str(CACHE_DIR),
        comp_ccm_key,
        "crsp_1960_2024_raw",
        f"{crsp_key}_ret",
        f"{crsp_key}_me",
        "ff_factors",
        f"{crsp_key}_nyse",
        q_key,
        fin_panel_key,
        sic_panel_key,
    )

    return dict(worker_initargs=worker_initargs)


# ── Signal loading ───────────────────────────────────────────────────────

def load_survivors(experiment_dirs: list[Path]) -> list[dict]:
    from siglab.lab.posthoc import collect_horse_race_survivors

    survivors = collect_horse_race_survivors(experiment_dirs)
    print(f"Loaded {len(survivors)} horse-race survivors from experiment archives")
    return survivors


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-model alpha tests")
    parser.add_argument("--experiment-dir", action="append")
    parser.add_argument("--experiment-dirs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
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

    survivors = load_survivors(experiment_dirs)
    if not survivors:
        print("No horse-race survivors found. Run horse-race selection first.")
        return
    completed_ids = _completed_signal_ids(out) if args.resume else set()
    if completed_ids:
        print(f"Resuming multi-model alpha: skipping {len(completed_ids)} completed signals")
        survivors = _pending_survivors(survivors, completed_ids)

    # Build tasks
    tasks = []
    for sig in survivors:
        excl_fin = bool(sig.get("exclude_financials", True))
        excl_micro = bool(sig.get("exclude_microcap", True))
        tasks.append((
            sig["signal_id"],
            sig["name"],
            sig["expression"],
            sig.get("expected_sign", "positive"),
            excl_fin,
            excl_micro,
        ))
    if not tasks:
        if args.resume:
            previous_results = _read_json_list(out)
            _write_json_list(out, previous_results)
            print(f"No pending multi-model alpha signals; preserved existing output -> {out}")
        else:
            print("No pending multi-model alpha signals.")
        return

    print("Loading data from cache...")
    data = load_data()

    n_evals = len(tasks) * len(MODELS)
    print(f"\n{len(tasks)} signals x {len(MODELS)} models = "
          f"{n_evals} evaluations — {args.workers} workers\n")

    result_map: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=data["worker_initargs"],
    ) as pool:
        futures = {pool.submit(_eval_multi_model_worker, t): t[:2] for t in tasks}
        with tqdm(total=len(futures), desc="Multi-model alpha", unit="sig") as pbar:
            for future in as_completed(futures):
                signal_id, name = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = {"signal_id": signal_id, "name": name, "results": {}, "error": str(e)}
                result_map[signal_id] = res

                sub = res.get("results", {})
                parts = []
                for m in MODELS:
                    mn = MODEL_NAMES[m]
                    t = sub.get(mn, {}).get("talpha")
                    parts.append(f"{mn}={t:.2f}" if t is not None else f"{mn}=err")
                tqdm.write(f"  {name:<35s} {' | '.join(parts)}")
                pbar.update(1)

    # Assemble output
    all_results = _assemble_rows(survivors, result_map)

    previous_results = _read_json_list(out) if args.resume else []
    rerun_ids = {str(sig.get("signal_id")) for sig in survivors}
    previous_results = [
        r for r in previous_results
        if str(r.get("signal_id")) not in rerun_ids
    ]
    all_results = previous_results + all_results
    _write_json_list(out, all_results)
    print(f"\nSaved → {out}")

    # Print summary table
    print(f"\n{'Signal':<35s} {'FF5 t':>7s} {'FF6 t':>7s} {'q4 t':>7s}")
    print("-" * 63)
    for row in all_results:
        parts = []
        for mn in ["FF5", "FF6", "q4"]:
            t = row.get(f"{mn}_talpha")
            parts.append(f"{t:7.2f}" if t is not None else "    err")
        print(f"  {row['name']:<33s} {''.join(parts)}")

    # Count survivors per model
    print(f"\nSurvivors by model (|t(alpha)| > 1.96):")
    for m in MODELS:
        mn = MODEL_NAMES[m]
        n_surv = sum(
            1 for r in all_results
            if r.get(f"{mn}_survives_196")
        )
        print(f"  {mn}: {n_surv}/{len(all_results)}")


if __name__ == "__main__":
    main()
