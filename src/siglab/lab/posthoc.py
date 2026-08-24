from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha1
import json
import math
from pathlib import Path
import re
import sys


DEFAULT_POSTHOC_SUITES = [
    "multiple_testing",
    "double_bootstrap",
    "horse_race",
    "multi_model_alpha",
    "subsample",
    "spanning",
]


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _iter_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _experiment_id(exp_dir: Path) -> str:
    manifest = _read_json(exp_dir / "manifest.json", {})
    return manifest.get("experiment_id") or exp_dir.name


def normalize_signal_expression(expression: object) -> str:
    """Normalize expression text for stable posthoc signal identity."""
    normalized = " ".join(str(expression).strip().lower().split())
    normalized = re.sub(r"\s*\(\s*", "( ", normalized)
    normalized = re.sub(r"\s*\)\s*", " )", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    return " ".join(normalized.split())


def signal_identity(signal: dict) -> tuple[str, str, str]:
    """Return the stable identity tuple for a discovered/evaluated signal."""
    source = signal.get("source_experiment_dir") or signal.get("source_experiment") or ""
    name = signal.get("display_name") or signal.get("name") or ""
    expression = signal.get("expression", "")
    return (
        str(source),
        str(name),
        normalize_signal_expression(expression),
    )


def signal_id_for(signal: dict) -> str:
    """Return a short stable id for a signal identity tuple."""
    payload = json.dumps(signal_identity(signal), separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()[:12]


def _with_signal_identity(signal: dict) -> dict:
    item = dict(signal)
    item["normalized_expression"] = normalize_signal_expression(item.get("expression", ""))
    item["signal_id"] = signal_id_for(item)
    return item


def _duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def validate_experiment_dirs(experiment_dirs: list[str | Path]) -> list[Path]:
    """Validate completed experiment folders used by posthoc tools."""
    paths = [Path(path) for path in experiment_dirs]
    errors = []
    for path in paths:
        if not path.is_dir():
            errors.append(f"{path} is not a directory")
        elif not (path / "archive.json").exists():
            errors.append(f"{path / 'archive.json'} is missing")
    if errors:
        raise ValueError("; ".join(errors))
    return paths


def collect_evaluated_signals(experiment_dirs: list[str | Path]) -> list[dict]:
    """Read evaluated signals from one or more experiment archive files."""
    rows: list[dict] = []
    for raw_dir in experiment_dirs:
        exp_dir = Path(raw_dir).resolve()
        archive = _read_json(exp_dir / "archive.json", {})
        exp_id = _experiment_id(exp_dir)
        for row in archive.get("evaluated", []):
            item = dict(row)
            item.setdefault("source_experiment", exp_id)
            item.setdefault("source_experiment_dir", str(exp_dir))
            rows.append(_with_signal_identity(item))
    return rows


def collect_considered_proposals(experiment_dirs: list[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for raw_dir in experiment_dirs:
        exp_dir = Path(raw_dir)
        archive = _read_json(exp_dir / "archive.json", {})
        exp_id = _experiment_id(exp_dir)
        for row in archive.get("considered", []):
            item = dict(row)
            item.setdefault("source_experiment", exp_id)
            item.setdefault("source_experiment_dir", str(exp_dir))
            rows.append(item)
    return rows


def signed_t_for_hypothesis(signal: dict) -> float | None:
    t_stat = signal.get("fmb_tstat")
    if t_stat is None:
        return None
    t_stat = float(t_stat)
    if not math.isfinite(t_stat):
        return None
    expected_sign = signal.get("expected_sign", "positive")
    if expected_sign == "negative":
        return -t_stat
    return t_stat


def collect_t_stats(experiment_dirs: list[str | Path]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    n_nonfinite = 0
    for signal in collect_evaluated_signals(experiment_dirs):
        if signal.get("error") is not None:
            continue
        signed_t = signed_t_for_hypothesis(signal)
        if signed_t is None:
            # error=None but no usable t (missing or NaN/inf): excluded so
            # the multiple-testing M reflects real tests, and counted so
            # the exclusion is visible.
            n_nonfinite += 1
            continue
        name = signal.get("name") or signal.get("expression")
        out.append((str(name), signed_t))
    if n_nonfinite:
        print(
            f"[posthoc] WARNING: excluded {n_nonfinite} evaluated signal(s) "
            f"with missing/non-finite fmb_tstat and error=None; "
            f"{len(out)} usable t-stats remain",
            file=sys.stderr,
        )
    return out


def _task_metadata(exp_dir: Path) -> dict:
    task_path = exp_dir / "task.json"
    if not task_path.exists():
        return {}
    return _read_json(task_path, {}).get("metadata", {})


def collect_first_pass_discoveries(
    experiment_dirs: list[str | Path],
    *,
    threshold: float = 1.96,
) -> list[dict]:
    """Collect significant signals with the expected empirical direction."""
    metadata_by_dir = {
        str(Path(path).resolve()): _task_metadata(Path(path).resolve())
        for path in experiment_dirs
    }
    out = []
    seen: set[tuple[str, str, str]] = set()
    for signal in collect_evaluated_signals(experiment_dirs):
        if signal.get("error") is not None:
            continue
        t_stat = signal.get("fmb_tstat")
        if t_stat is None or not math.isfinite(float(t_stat)):
            continue
        if abs(float(t_stat)) <= threshold:
            continue
        if not _is_correct_sign(signal):
            continue
        key = signal_identity(signal)
        if key in seen:
            continue
        seen.add(key)
        source_dir = signal["source_experiment_dir"]
        metadata = metadata_by_dir.get(source_dir, {})
        row = dict(signal)
        row["pair_name"] = metadata.get("pair_name", metadata.get("pair"))
        row["exclude_financials"] = bool(metadata.get("exclude_financials", True))
        row["exclude_microcap"] = bool(metadata.get("exclude_microcap", True))
        out.append(row)
    return out


def collect_successful_signals(
    experiment_dirs: list[str | Path],
    *,
    threshold: float = 1.96,
) -> list[dict]:
    """Compatibility wrapper for first-pass discovery survivors."""
    return collect_first_pass_discoveries(experiment_dirs, threshold=threshold)


def collect_horse_race_survivors(experiment_dirs: list[str | Path]) -> list[dict]:
    """Collect survivor rows from explicit horse-race selection artifacts."""
    out = []
    for raw_dir in experiment_dirs:
        exp_dir = Path(raw_dir).resolve()
        artifact_path = exp_dir / "selection" / "horse_race.json"
        if not artifact_path.exists():
            raise ValueError(
                f"{artifact_path} is missing; run_horse_race_selection.py should "
                "be run first"
            )
        artifact = _read_json(artifact_path, {})
        if artifact.get("selection") != "horse_race":
            raise ValueError(f"{artifact_path} must have selection='horse_race'")

        evaluated = collect_evaluated_signals([exp_dir])
        evaluated_ids = [row["signal_id"] for row in evaluated]
        duplicate_evaluated_ids = _duplicate_values(evaluated_ids)
        if duplicate_evaluated_ids:
            raise ValueError(
                f"{artifact_path} duplicate evaluated archive signal_id values: "
                f"{duplicate_evaluated_ids}"
            )

        evaluated_by_id = {row["signal_id"]: row for row in evaluated}
        input_signal_rows = artifact.get("input_signals", [])
        if not isinstance(input_signal_rows, list):
            raise ValueError(
                f"{artifact_path} input_signals must be a list of rows with signal_id"
            )

        input_signal_ids = []
        for index, row in enumerate(input_signal_rows):
            signal_id = row.get("signal_id") if isinstance(row, dict) else None
            if not isinstance(signal_id, str) or not signal_id.strip():
                raise ValueError(
                    f"{artifact_path} input_signals[{index}] must be a dict with "
                    "a non-empty signal_id"
                )
            input_signal_ids.append(signal_id)
        duplicate_input_ids = _duplicate_values(input_signal_ids)
        if duplicate_input_ids:
            raise ValueError(
                f"{artifact_path} duplicate input_signals signal_id values: "
                f"{duplicate_input_ids}"
            )

        # P2-23: artifact rows must prove their own identity. Recompute the id
        # from each row's content (source / name / normalized expression); a
        # stored id that does not match its own row means the artifact was
        # hand-edited or went stale after writing.
        id_mismatches = sorted(
            row["signal_id"]
            for row in input_signal_rows
            if signal_id_for(row) != row["signal_id"]
        )
        if id_mismatches:
            raise ValueError(
                f"{artifact_path} input_signals rows failed identity "
                f"recomputation (stored signal_id != signal_id_for(row)): "
                f"{id_mismatches}; regenerate the artifact with "
                "run_horse_race_selection.py instead of editing it"
            )

        survivor_ids = list(artifact.get("stepwise_survivors", []))
        duplicate_survivor_ids = _duplicate_values(survivor_ids)
        if duplicate_survivor_ids:
            raise ValueError(
                f"{artifact_path} duplicate stepwise_survivors signal_id values: "
                f"{duplicate_survivor_ids}"
            )

        input_ids = set(input_signal_ids)
        input_by_id = {
            row["signal_id"]: row
            for row in input_signal_rows
        }
        archive_ids = set(evaluated_by_id)

        missing_inputs = input_ids - archive_ids
        if missing_inputs:
            raise ValueError(
                f"{artifact_path} input_signals not found in evaluated archive: "
                f"{sorted(missing_inputs)}"
            )

        missing_from_inputs = set(survivor_ids) - input_ids
        if missing_from_inputs:
            raise ValueError(
                f"{artifact_path} stepwise_survivors not found in input_signals: "
                f"{sorted(missing_from_inputs)}"
            )

        missing_from_archive = set(survivor_ids) - archive_ids
        if missing_from_archive:
            raise ValueError(
                f"{artifact_path} stepwise_survivors not found in evaluated archive: "
                f"{sorted(missing_from_archive)}"
            )

        # For survivors, the artifact's raw expression must equal the archive
        # row's raw expression — normalization-invariant drift is still drift.
        expression_mismatches = sorted(
            signal_id
            for signal_id in survivor_ids
            if str(input_by_id[signal_id].get("expression", ""))
            != str(evaluated_by_id[signal_id].get("expression", ""))
        )
        if expression_mismatches:
            raise ValueError(
                f"{artifact_path} stepwise_survivors whose artifact expression "
                f"differs from the evaluated-archive expression: "
                f"{expression_mismatches}; the artifact predates an archive "
                "change — regenerate it with run_horse_race_selection.py"
            )

        stepwise_t = artifact.get("stepwise_t", {})
        if not isinstance(stepwise_t, dict):
            raise ValueError(f"{artifact_path} stepwise_t must be a dict")
        for signal_id in survivor_ids:
            if signal_id not in stepwise_t or stepwise_t[signal_id] is None:
                raise ValueError(
                    f"{artifact_path} stepwise_t missing numeric value for {signal_id}"
                )
            try:
                horse_race_t = float(stepwise_t[signal_id])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{artifact_path} stepwise_t value for {signal_id} must be numeric"
                ) from exc
            row = dict(input_by_id[signal_id])
            row["horse_race_t"] = horse_race_t
            row["selection"] = "horse_race"
            out.append(row)
    return out


def _is_correct_sign(signal: dict) -> bool:
    t_stat = signal.get("fmb_tstat")
    if t_stat is None or not math.isfinite(float(t_stat)):
        return False
    expected_positive = signal.get("expected_sign", "positive") == "positive"
    actual_positive = float(t_stat) > 0
    return expected_positive == actual_positive


def _summarize_one_experiment(exp_dir: Path) -> dict:
    archive = _read_json(exp_dir / "archive.json", {})
    considered = archive.get("considered", [])
    evaluated = archive.get("evaluated", [])
    diagnostics = _iter_jsonl(exp_dir / "validation_diagnostics.jsonl")
    parser_diagnostics = _iter_jsonl(exp_dir / "parser_diagnostics.jsonl")
    rejected = [row for row in diagnostics if row.get("code") != "accepted"]
    rejected.extend(
        row for row in parser_diagnostics
        if str(row.get("severity", "")) in {"error", "warning"}
    )
    rejection_codes = Counter(str(row.get("code", "unknown")) for row in rejected)

    eval_errors = [row for row in evaluated if row.get("error") is not None]
    no_error = [row for row in evaluated if row.get("error") is None]
    # error=None but no finite t previously fell into NO bucket, so the
    # per-experiment buckets did not sum to evaluated_tests.
    invalid_results = [
        row for row in no_error
        if row.get("fmb_tstat") is None
        or not math.isfinite(float(row["fmb_tstat"]))
    ]
    finite = [
        row for row in no_error
        if row.get("fmb_tstat") is not None
        and math.isfinite(float(row["fmb_tstat"]))
    ]
    wrong_sign = [
        row for row in finite
        if abs(float(row["fmb_tstat"])) > 1.96 and not _is_correct_sign(row)
    ]
    significant_correct = [
        row for row in finite
        if abs(float(row["fmb_tstat"])) > 1.96 and _is_correct_sign(row)
    ]
    weak = [
        row for row in finite
        if abs(float(row["fmb_tstat"])) <= 1.96
    ]

    return {
        "experiment_id": _experiment_id(exp_dir),
        "experiment_dir": str(exp_dir),
        "considered_proposals": len(considered),
        "unique_considered_expressions": len({
            str(row.get("expression", "")).strip().lower()
            for row in considered
            if row.get("expression")
        }),
        "evaluated_tests": len(evaluated),
        "evaluation_errors": len(eval_errors),
        "invalid_results": len(invalid_results),
        "wrong_sign": len(wrong_sign),
        "weak_signals": len(weak),
        "significant_correct_sign": len(significant_correct),
        "validation_rejections": len(rejected),
        "validation_rejection_codes": dict(sorted(rejection_codes.items())),
    }


def build_search_universe_summary(
    experiment_dirs: list[str | Path],
) -> dict:
    experiments = [_summarize_one_experiment(Path(path)) for path in experiment_dirs]
    rejection_codes: Counter[str] = Counter()
    for exp in experiments:
        rejection_codes.update(exp["validation_rejection_codes"])

    fields = [
        "considered_proposals",
        "unique_considered_expressions",
        "evaluated_tests",
        "evaluation_errors",
        "invalid_results",
        "wrong_sign",
        "weak_signals",
        "significant_correct_sign",
        "validation_rejections",
    ]
    totals = {"experiments": len(experiments)}
    for field in fields:
        totals[field] = sum(int(exp[field]) for exp in experiments)

    return {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals": totals,
        "validation_rejection_codes": dict(sorted(rejection_codes.items())),
        "experiments": experiments,
    }


def write_search_universe_summary(
    experiment_dirs: list[str | Path],
    output_path: str | Path,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            build_search_universe_summary(experiment_dirs),
            indent=2,
            sort_keys=True,
        )
    )
    return out


def write_posthoc_manifest(
    *,
    experiment_dir: str | Path,
    suites: list[str] | None = None,
    output_root: str | Path | None = None,
) -> Path:
    exp_dir = Path(experiment_dir)
    posthoc_dir = exp_dir / "posthoc"
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    suites = suites or list(DEFAULT_POSTHOC_SUITES)
    aggregate_root = Path(output_root) if output_root is not None else None

    payload = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_dir": str(exp_dir),
        "inputs": {
            "manifest": str(exp_dir / "manifest.json"),
            "task": str(exp_dir / "task.json"),
            "archive": str(exp_dir / "archive.json"),
            "search_universe": str(exp_dir / "search_universe.json"),
        },
        "suites": [
            {
                "name": suite,
                "status": "pending",
                "output_dir": str(
                    exp_dir / "selection"
                    if suite == "horse_race"
                    else (
                        aggregate_root / suite
                        if aggregate_root is not None
                        else posthoc_dir / suite
                    )
                ),
            }
            for suite in suites
        ],
    }

    out = posthoc_dir / "posthoc_manifest.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out
