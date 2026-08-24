"""Multiple testing & Bayesian inference for tested signals.

Applies HLZ (2016) multiple testing corrections (Bonferroni, Holm, BHY)
and Harvey (2017) Bayesian inference (MBF, Bayesianized p-values) to
the full set of tested signals.

No data loading or signal evaluation needed. This reads completed experiment
archives and uses the observed evaluated-test universe as M.

Usage
-----
    python scripts/experiments/run_multiple_testing.py \
        --experiment-dir data/experiments/<timestamp>_<run>_<pair>
    python scripts/experiments/run_multiple_testing.py \
        --experiment-dirs data/experiments/<exp-a> data/experiments/<exp-b>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from siglab.assay.sample import POSTHOC_RESULTS_SEMANTICS

def _experiment_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = []
    if args.experiment_dir:
        dirs.extend(args.experiment_dir)
    if args.experiment_dirs:
        dirs.extend(args.experiment_dirs)
    return [Path(path) for path in dirs]


def _default_output_path(experiment_dirs: list[Path]) -> Path:
    if len(experiment_dirs) == 1:
        return experiment_dirs[0] / "posthoc" / "multiple_testing" / "results.json"
    return (
        experiment_dirs[0].parent
        / "posthoc_aggregate"
        / "multiple_testing"
        / "results.json"
    )


def main():
    parser = argparse.ArgumentParser(description="Multiple testing & Bayesian inference")
    parser.add_argument(
        "--experiment-dir",
        action="append",
        help="Completed experiment folder. Can be provided multiple times.",
    )
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        help="Completed experiment folders.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--print-limit",
        type=int,
        default=50,
        help="Maximum Bayesian rows to print to stdout; saved JSON always contains all rows.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from siglab.assay.multiple_testing import (
        multiple_testing_summary,
        bayesian_report,
        required_t_for_bayesian_threshold,
    )

    # ── Collect all t-stats ──────────────────────────────────────────────
    experiment_dirs = _experiment_dirs(args)
    if not experiment_dirs:
        parser.error("provide --experiment-dir or --experiment-dirs")

    from siglab.lab.posthoc import (
        collect_t_stats,
        validate_experiment_dirs,
    )

    try:
        experiment_dirs = validate_experiment_dirs(experiment_dirs)
    except ValueError as exc:
        parser.error(str(exc))
    t_stats = collect_t_stats(experiment_dirs)
    print(f"Collected {len(t_stats)} evaluated tests with t-stats")

    if not t_stats:
        print("No results found. Run discovery first, then point posthoc tools at artifacts.")
        return

    # ── Part 1: Multiple Testing ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PART 1: MULTIPLE TESTING (M={len(t_stats)}, alpha={args.alpha})")
    print(f"{'='*70}")

    summary = multiple_testing_summary(t_stats, alpha=args.alpha)

    print(f"\n  |t| hurdles:")
    print(f"    Bonferroni (FWER): |t| > {summary.bonf_threshold_t:.3f}")
    print(f"    Holm       (FWER): |t| > {summary.holm_threshold_t:.3f}")
    print(f"    BHY        (FDR):  |t| > {summary.bhy_threshold_t:.3f}")

    print(f"\n  Rejections (out of {summary.M} signals):")
    print(f"    Bonferroni: {summary.n_reject_bonf}")
    print(f"    Holm:       {summary.n_reject_holm}")
    print(f"    BHY:        {summary.n_reject_bhy}")

    # ── Part 2: Bayesian Inference ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PART 2: BAYESIAN INFERENCE (Harvey 2017)")
    print(f"{'='*70}")

    # Required t-stat thresholds
    print(f"\n  Required |t| for Pr(H0|data) <= 5%:")
    for pi_label, pi in [("5% (skeptical)", 0.05), ("20% (moderate)", 0.20), ("50% (agnostic)", 0.50)]:
        t_mbf = required_t_for_bayesian_threshold(pi, 0.05, "mbf")
        t_sd = required_t_for_bayesian_threshold(pi, 0.05, "sd_mbf")
        print(f"    pi_0={pi_label:<20s}  MBF: |t|>{t_mbf:.2f}   SD-MBF: |t|>{t_sd:.2f}")

    # Bayesian report for the full evaluated universe
    bayes = bayesian_report(t_stats)

    sorted_bayes = sorted(bayes, key=lambda x: abs(x.t_stat), reverse=True)
    if args.print_limit >= 0:
        printable_bayes = sorted_bayes[:args.print_limit]
        print(
            f"\n  Showing top {len(printable_bayes)} of {len(sorted_bayes)} "
            "evaluated signals by |t|"
        )
    else:
        printable_bayes = sorted_bayes
        print(f"\n  Showing all {len(sorted_bayes)} evaluated signals by |t|")

    print(f"\n  {'Signal':<35s} {'|t|':>5s} {'MBF':>8s} {'Pr(H0) 5%':>10s} {'Pr(H0) 20%':>10s} {'Pr(H0) 50%':>10s}")
    print("  " + "-" * 80)
    for r in printable_bayes:
        print(f"  {r.name:<35s} {abs(r.t_stat):5.2f} {r.mbf:8.5f} "
              f"{r.bayes_p['bayes_p_005']:10.4f} {r.bayes_p['bayes_p_020']:10.4f} "
              f"{r.bayes_p['bayes_p_050']:10.4f}")

    # Count evaluated signals by Bayesian criterion
    for pi_label, key in [("5%", "bayes_p_005"), ("20%", "bayes_p_020"), ("50%", "bayes_p_050")]:
        n = sum(1 for r in bayes if r.bayes_p[key] < 0.05)
        print(f"\n  Pr(H0|data)<5% under pi_0={pi_label}: {n}/{len(bayes)} evaluated signals")

    # ── Save combined results ────────────────────────────────────────────
    from dataclasses import asdict

    output = {
        "semantics": POSTHOC_RESULTS_SEMANTICS,
        "multiple_testing": {
            "M": summary.M,
            "alpha": summary.alpha,
            "bonf_threshold_t": summary.bonf_threshold_t,
            "holm_threshold_t": summary.holm_threshold_t,
            "bhy_threshold_t": summary.bhy_threshold_t,
            "n_reject_bonf": summary.n_reject_bonf,
            "n_reject_holm": summary.n_reject_holm,
            "n_reject_bhy": summary.n_reject_bhy,
            "per_signal": summary.results,
        },
        "bayesian": {
            "population": "full_evaluated_universe",
            "n_signals": len(bayes),
            "signals": [asdict(r) for r in bayes],
        },
    }

    out = Path(args.output) if args.output else _default_output_path(experiment_dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    from siglab.utils.json import write_strict_json

    write_strict_json(out, output, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
