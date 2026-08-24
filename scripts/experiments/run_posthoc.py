"""Prepare and optionally execute posthoc suites for completed experiments."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from siglab.lab.posthoc import (
    DEFAULT_POSTHOC_SUITES,
    validate_experiment_dirs,
    write_posthoc_manifest,
    write_search_universe_summary,
)


SUITE_SCRIPTS = {
    "multiple_testing": "run_multiple_testing.py",
    "double_bootstrap": "run_double_bootstrap.py",
    "horse_race": "run_horse_race_selection.py",
    "spanning": "run_spanning_test.py",
    "multi_model_alpha": "run_multi_model_alpha.py",
    "subsample": "run_subsample.py",
}

ROBUSTNESS_SUITES = {
    "multi_model_alpha",
    "subsample",
    "spanning",
}

DEFAULT_SUITE_WORKERS = {
    "double_bootstrap": 12,
    "horse_race": 4,
    "multi_model_alpha": 4,
    "subsample": 4,
    "spanning": 6,
}

SUITE_WORKER_DESTS = {
    "double_bootstrap": "double_bootstrap_workers",
    "horse_race": "horse_race_workers",
    "multi_model_alpha": "multi_model_alpha_workers",
    "subsample": "subsample_workers",
    "spanning": "spanning_workers",
}


def _experiment_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = []
    if args.experiment_dir:
        dirs.extend(args.experiment_dir)
    if args.experiment_dirs:
        dirs.extend(args.experiment_dirs)
    return [Path(path) for path in dirs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and run posthoc suites")
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
    parser.add_argument(
        "--aggregate-output",
        help=(
            "Optional path for aggregate search_universe.json across all inputs. "
            "Defaults below --output-root when that option is provided."
        ),
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Root for aggregate suite artifacts; each suite writes "
            "<root>/<suite>/results.json. Horse-race artifacts remain inside "
            "each experiment directory."
        ),
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=DEFAULT_POSTHOC_SUITES,
    )
    parser.add_argument(
        "--run-suites",
        action="store_true",
        help="Execute the requested posthoc suite scripts after preparation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Legacy common worker override for horse-race and robustness suites. "
            "A suite-specific worker option takes precedence."
        ),
    )
    parser.add_argument("--double-bootstrap-workers", type=int, default=None)
    parser.add_argument(
        "--double-bootstrap-beta-workers",
        type=int,
        choices=range(1, 7),
        default=4,
    )
    parser.add_argument("--horse-race-workers", type=int, default=None)
    parser.add_argument("--multi-model-alpha-workers", type=int, default=None)
    parser.add_argument("--subsample-workers", type=int, default=None)
    parser.add_argument("--spanning-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-beta-cache", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--multiple-testing-alpha", type=float, default=0.05)
    parser.add_argument("--multiple-testing-print-limit", type=int, default=50)
    parser.add_argument(
        "--double-bootstrap-signal-scope",
        choices=["evaluated", "first_pass"],
        default="evaluated",
    )
    parser.add_argument("--double-bootstrap-p0", type=float, default=0.10)
    parser.add_argument(
        "--double-bootstrap-I",
        "--double-bootstrap-i",
        dest="double_bootstrap_i",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--double-bootstrap-J",
        "--double-bootstrap-j",
        dest="double_bootstrap_j",
        type=int,
        default=200,
    )
    parser.add_argument("--double-bootstrap-nw-lags", type=int, default=6)
    parser.add_argument("--horse-race-threshold", type=float, default=1.96)
    parser.add_argument("--spanning-top-k", type=int, default=10)
    parser.add_argument("--spanning-corr-sample-every", type=int, default=24)
    return parser


def _ordered_suites(suites: list[str]) -> list[str]:
    unknown = sorted(set(suites) - set(SUITE_SCRIPTS))
    if unknown:
        known = ", ".join(sorted(SUITE_SCRIPTS))
        raise ValueError(
            f"unknown posthoc suite(s) {unknown}; expected one of {known}"
        )
    requested = set(suites)
    return [suite for suite in DEFAULT_POSTHOC_SUITES if suite in requested]


def _suite_workers(suite: str, args: argparse.Namespace) -> int | None:
    dest = SUITE_WORKER_DESTS.get(suite)
    if dest is None:
        return None
    suite_value = getattr(args, dest, None)
    if suite_value is not None:
        return int(suite_value)
    if suite != "double_bootstrap" and args.workers is not None:
        return int(args.workers)
    return DEFAULT_SUITE_WORKERS[suite]


def _suite_output_path(suite: str, args: argparse.Namespace) -> Path | None:
    if not args.output_root or suite == "horse_race":
        return None
    return Path(args.output_root) / suite / "results.json"


def _suite_command(
    suite: str,
    experiment_dirs: list[Path],
    args: argparse.Namespace | None = None,
) -> list[str]:
    script = SUITE_SCRIPTS.get(suite)
    if script is None:
        known = ", ".join(sorted(SUITE_SCRIPTS))
        raise ValueError(f"unknown posthoc suite {suite!r}; expected one of {known}")

    cmd = [sys.executable, str(Path(__file__).with_name(script))]
    if len(experiment_dirs) == 1:
        cmd.extend(["--experiment-dir", str(experiment_dirs[0])])
    else:
        cmd.append("--experiment-dirs")
        cmd.extend(str(path) for path in experiment_dirs)

    if args is None:
        return cmd

    output_path = _suite_output_path(suite, args)
    if output_path is not None:
        cmd.extend(["--output", str(output_path)])

    workers = _suite_workers(suite, args)

    if suite == "multiple_testing":
        cmd.extend(["--alpha", str(args.multiple_testing_alpha)])
        cmd.extend(["--print-limit", str(args.multiple_testing_print_limit)])

    if suite == "double_bootstrap":
        cmd.extend(["--signal-scope", args.double_bootstrap_signal_scope])
        cmd.extend(["--p0", str(args.double_bootstrap_p0)])
        cmd.extend(["--I", str(args.double_bootstrap_i)])
        cmd.extend(["--J", str(args.double_bootstrap_j)])
        cmd.extend(["--nw-lags", str(args.double_bootstrap_nw_lags)])
        cmd.extend(["--n-workers", str(workers)])
        cmd.extend([
            "--beta-workers",
            str(args.double_bootstrap_beta_workers),
        ])
        if args.resume_beta_cache:
            cmd.append("--resume-beta-cache")

    if suite == "horse_race":
        cmd.extend(["--threshold", str(args.horse_race_threshold)])

    if suite in {
        "horse_race",
        "spanning",
        "multi_model_alpha",
        "subsample",
    }:
        cmd.extend(["--workers", str(workers)])

    if suite == "horse_race" and args.skip_existing:
        cmd.append("--skip-existing")

    if suite in {"spanning", "multi_model_alpha", "subsample"} and args.resume:
        cmd.append("--resume")

    if suite == "spanning":
        cmd.extend(["--top-k", str(args.spanning_top_k)])
        cmd.extend(["--corr-sample-every", str(args.spanning_corr_sample_every)])

    return cmd


def _missing_selection_artifacts(experiment_dirs: list[Path]) -> list[Path]:
    return [
        exp_dir / "selection" / "horse_race.json"
        for exp_dir in experiment_dirs
        if not (exp_dir / "selection" / "horse_race.json").exists()
    ]


def _run_suites(
    suites: list[str],
    experiment_dirs: list[Path],
    args: argparse.Namespace,
) -> None:
    selection_ready = not _missing_selection_artifacts(experiment_dirs)
    for suite in _ordered_suites(suites):
        if suite in ROBUSTNESS_SUITES and not selection_ready:
            missing = _missing_selection_artifacts(experiment_dirs)
            raise ValueError(
                "robustness suites require completed horse-race selection; "
                f"missing artifacts: {[str(path) for path in missing]}"
            )
        cmd = _suite_command(suite, experiment_dirs, args)
        print(f"\nRunning posthoc suite: {suite}")
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        if suite == "horse_race":
            missing = _missing_selection_artifacts(experiment_dirs)
            if missing:
                raise ValueError(
                    "horse-race suite completed without writing all selection "
                    f"artifacts: {[str(path) for path in missing]}"
                )
            selection_ready = True


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
    try:
        suites = _ordered_suites(args.suites)
    except ValueError as exc:
        parser.error(str(exc))
    suite_output_root = (
        Path(args.output_root)
        if args.output_root
        else (
            experiment_dirs[0].parent / "posthoc_aggregate"
            if len(experiment_dirs) > 1
            else None
        )
    )

    for exp_dir in experiment_dirs:
        search_out = write_search_universe_summary(
            [exp_dir],
            exp_dir / "search_universe.json",
        )
        manifest_out = write_posthoc_manifest(
            experiment_dir=exp_dir,
            suites=suites,
            output_root=suite_output_root,
        )
        print(f"Wrote {search_out}")
        print(f"Wrote {manifest_out}")

    if suite_output_root is not None or args.aggregate_output:
        aggregate_out = (
            Path(args.aggregate_output)
            if args.aggregate_output
            else suite_output_root / "search_universe.json"
        )
        write_search_universe_summary(experiment_dirs, aggregate_out)
        print(f"Wrote {aggregate_out}")

    if args.run_suites:
        try:
            _run_suites(suites, experiment_dirs, args)
        except ValueError as exc:
            parser.error(str(exc))
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
