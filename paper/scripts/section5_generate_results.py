#!/usr/bin/env python3
"""Generate the Section 5 figures and table block from formal run artifacts.

The script treats all six executions as separate search and testing families.
Primary results are summarized by the median and full range of B1--B3; the
three comparison configurations remain single, descriptive realizations.

Outputs
-------
paper/figures/fig1_generation_dynamics.pdf and .png
paper/figures/fig2_pair_frontier.pdf and .png
paper/figures/fig3_architecture_contrasts.pdf and .png
paper/tables.tex (the tagged Section 5 block only)

Run from any working directory:

    python paper/scripts/section5_generate_results.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
FIGURE_DIR = PAPER_DIR / "figures"
GENERATED_DIR = PAPER_DIR / "generated"
FORMAL_DIR = PAPER_DIR / "data" / "formal"
POSTHOC_DIR = FORMAL_DIR / "posthoc"
MECHANISM_CODES = GENERATED_DIR / "mechanism_codes.csv"
MECHANISM_AGREEMENT = GENERATED_DIR / "mechanism_coding_agreement.json"

sys.path.insert(0, str(REPO_ROOT / "src"))
from siglab.agent.variables import VARIABLE_CATALOG  # noqa: E402

THEME_KEYS = {
    "profit": "Profitability",
    "value": "Valuation",
    "invest": "Investment",
    "accrual": "Accruals",
    "quality": "Quality",
    "finance": "Financing",
    "intangible": "Intangibles",
    "distress": "Distress",
}
_TOKEN_RX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def variable_set(expression: object) -> frozenset:
    """Catalog variables used by an expression (operators are upper-case, skipped)."""
    return frozenset(tok for tok in _TOKEN_RX.findall(str(expression)) if tok in VARIABLE_CATALOG)

RUNS = [
    ("B1", "Primary B1", "baseline_r1", True),
    ("B2", "Primary B2", "baseline_r2", True),
    ("B3", "Primary B3", "baseline_r3", True),
    ("A1", "Single agent", "ablation_no_critique", False),
    ("A2", "GPT-5.5 proposer", "ablation_gpt_5_5", False),
    ("A3", "GPT-5.6-sol proposer", "ablation_gpt_5_6_sol", False),
]
PRIMARY_KEYS = ["B1", "B2", "B3"]

COLORS = {
    "B1": "#6F91A8",
    "B2": "#315F78",
    "B3": "#9DB6C5",
    "A1": "#B24C3D",
    "A2": "#7A5AA6",
    "A3": "#D18B2C",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def tex(value: str) -> str:
    return (
        value.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def oriented(value: float | None, expected_sign: str) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value if expected_sign == "positive" else -value


def normalize_signal_expression(expression: object) -> str:
    """Match the stable expression normalization used by post-search tools."""
    normalized = " ".join(str(expression).strip().lower().split())
    normalized = re.sub(r"\s*\(\s*", "( ", normalized)
    normalized = re.sub(r"\s*\)\s*", " )", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    return " ".join(normalized.split())


def is_correct_sign(item: dict) -> bool:
    value = item.get("fmb_tstat")
    if value is None or not math.isfinite(value):
        return False
    return oriented(value, item.get("expected_sign", "positive")) > 1.96


def pair_metadata(experiment_dir: Path) -> tuple[str, str]:
    metadata = read_json(experiment_dir / "task.json")["metadata"]
    return metadata["pair_name"], metadata["interaction_label"]


def iter_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def rankdata(values: list[float]) -> np.ndarray:
    """Average ranks, matching Spearman's treatment of ties."""
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values_array[order[stop]] == values_array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def display_pair(value: str) -> str:
    labels = {
        "profit": "Profitability",
        "value": "Valuation",
        "invest": "Investment",
        "accrual": "Accruals",
        "quality": "Quality",
        "finance": "Financing",
        "intangible": "Intangibles",
        "distress": "Distress",
    }
    a, b = value.split("_", 1)
    return f"{labels[a]} x {labels[b]}"


def readable_signal(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def collect_run(key: str, label: str, dirname: str, primary: bool) -> dict:
    aggregate = read_json(POSTHOC_DIR / dirname / "search_universe.json")
    root = FORMAL_DIR / dirname
    pair_rows = {}
    generation = defaultdict(lambda: Counter(evaluated=0, correct=0))
    best_t = defaultdict(float)
    rejections = Counter()
    rejections_gen0 = Counter()
    evaluated_varsets = set()
    survivor_varsets = set()
    calls = Counter()
    output_tokens = 0
    critic_flags = 0
    evaluated_expressions = set()
    survivor_expressions = set()
    survivor_records = []

    for item in aggregate["experiments"]:
        # Aggregates created on another operating system can retain an absolute
        # source path. The experiment id is portable and the frozen run root is
        # authoritative in this repository.
        experiment_dir = root / item["experiment_id"]
        pair_name, pair_label = pair_metadata(experiment_dir)
        horse = read_json(experiment_dir / "selection" / "horse_race.json")
        pair_rows[pair_name] = {
            "pair_name": pair_name,
            "pair_label": pair_label,
            "considered": item["considered_proposals"],
            "evaluated": item["evaluated_tests"],
            "correct": item["significant_correct_sign"],
            "survivors": horse["n_survivors"],
        }

        archive = read_json(experiment_dir / "archive.json")
        for candidate in archive["evaluated"]:
            gen = int(candidate["generation"])
            generation[gen]["evaluated"] += 1
            hit = is_correct_sign(candidate)
            generation[gen]["correct"] += int(hit)
            if hit:
                best_t[gen] = max(best_t[gen], abs(candidate["fmb_tstat"]))
            normalized = normalize_signal_expression(candidate.get("expression", ""))
            if normalized:
                evaluated_expressions.add(normalized)
            evaluated_varsets.add(variable_set(candidate.get("expression", "")))

        for record in iter_jsonl(experiment_dir / "validation_diagnostics.jsonl"):
            code = record.get("code")
            if code and code != "accepted":
                rejections[code] += 1
                if int(record.get("generation", -1)) == 0:
                    rejections_gen0[code] += 1

        survivor_ids = set(horse["stepwise_survivors"])
        stepwise_t = horse.get("stepwise_t", {})
        for candidate in horse["input_signals"]:
            signal_id = candidate["signal_id"]
            if signal_id not in survivor_ids:
                continue
            normalized = candidate.get("normalized_expression")
            if normalized:
                survivor_expressions.add(normalized)
            survivor_varsets.add(variable_set(candidate.get("expression", "")))
            row = dict(candidate)
            row["stepwise_t"] = stepwise_t.get(signal_id)
            survivor_records.append(row)

        for raw_file in experiment_dir.glob("generations/gen_*/raw_responses.jsonl"):
            for response in iter_jsonl(raw_file):
                role = response.get("role", "unknown")
                calls[role] += 1
                output_tokens += int(response.get("output_tokens") or 0)
        for note_file in experiment_dir.glob("generations/gen_*/critic_notes.json"):
            for note in read_json(note_file):
                verdict = str(note.get("verdict", "")).lower()
                critic_flags += int(verdict not in {"pass", "accept", "accepted", "ok"})

    totals = aggregate["totals"]
    assert totals["experiments"] == 28
    assert len(pair_rows) == 28
    assert sum(row["survivors"] for row in pair_rows.values()) == len(survivor_records)
    assert sum(calls.values()) in {280, 839, 840}
    generation_rows = []
    for gen in range(10):
        counter = generation[gen]
        generation_rows.append(
            {
                "generation": gen,
                "evaluated": counter["evaluated"],
                "correct": counter["correct"],
                "rate": counter["correct"] / counter["evaluated"],
                "best_t": best_t[gen],
            }
        )

    return {
        "key": key,
        "label": label,
        "dirname": dirname,
        "primary": primary,
        "totals": totals,
        "pair_rows": pair_rows,
        "generation": generation_rows,
        "calls": dict(calls),
        "n_calls": sum(calls.values()),
        "output_tokens": output_tokens,
        "critic_flags": critic_flags,
        "evaluated_expressions": evaluated_expressions,
        "survivor_expressions": survivor_expressions,
        "evaluated_varsets": evaluated_varsets,
        "survivor_varsets": survivor_varsets,
        "rejections": dict(rejections),
        "rejections_gen0": dict(rejections_gen0),
        "survivor_records": survivor_records,
        "n_survivors": len(survivor_records),
    }


def load_mechanism_analysis(runs: dict[str, dict]) -> dict | None:
    """Load the blind codes once and validate their link to discovery records."""
    if not MECHANISM_CODES.exists():
        return None

    required = {
        "run", "pair_name", "signal_name", "expression", "primary_mechanism",
        "expected_sign", "record_id",
    }
    with MECHANISM_CODES.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None and required <= set(reader.fieldnames)
        rows = list(reader)

    expected_total = sum(runs[key]["totals"]["evaluated_tests"] for key, *_ in RUNS)
    assert len(rows) == expected_total == 9656
    assert len({row["record_id"] for row in rows}) == len(rows)

    run_counts = Counter(row["run"] for row in rows)
    assert set(run_counts) == {key for key, *_ in RUNS}
    for key, *_ in RUNS:
        assert run_counts[key] == runs[key]["totals"]["evaluated_tests"]

    survivor_keys = {
        key: {(row["pair_name"], row["name"]) for row in runs[key]["survivor_records"]}
        for key, *_ in RUNS
    }
    for key, *_ in RUNS:
        assert len(survivor_keys[key]) == runs[key]["n_survivors"]

    populations = {
        "evaluated": defaultdict(set),
        "survivors": defaultdict(set),
    }
    frequencies = defaultdict(Counter)
    matched_survivors = Counter()
    other_count = 0
    for row in rows:
        run = row["run"]
        pair_name = row["pair_name"]
        assert pair_name in runs[run]["pair_rows"]
        mechanism = row["primary_mechanism"].strip().lower()
        expected_sign = row["expected_sign"].strip().lower()
        assert mechanism and expected_sign in {"positive", "negative"}
        label = (mechanism, expected_sign)
        populations["evaluated"][(run, pair_name)].add(label)
        frequencies[(run, pair_name)][mechanism] += 1
        other_count += int(mechanism == "other")
        if (pair_name, row["signal_name"]) in survivor_keys[run]:
            populations["survivors"][(run, pair_name)].add(label)
            matched_survivors[run] += 1

    for key, *_ in RUNS:
        assert matched_survivors[key] == runs[key]["n_survivors"]
        for pair_name in runs[key]["pair_rows"]:
            assert populations["evaluated"][(key, pair_name)]
            assert populations["survivors"][(key, pair_name)]

    agreement = read_json(MECHANISM_AGREEMENT)
    assert agreement["n_records"] == len(rows)
    assert agreement["cohen_kappa"] >= 0.6
    other_share = other_count / len(rows)
    assert other_share <= 0.15

    diversity = {}
    for key, *_ in RUNS:
        for pair_name in sorted(runs[key]["pair_rows"]):
            counts = Counter(frequencies[(key, pair_name)])
            counts.pop("other", None)
            total = sum(counts.values())
            assert total > 0
            diversity[(key, pair_name)] = {
                "count": len(counts),
                "hhi": sum((count / total) ** 2 for count in counts.values()),
                "n_coded": total,
            }

    jaccard = {}
    for left_index, left in enumerate(PRIMARY_KEYS):
        for right in PRIMARY_KEYS[left_index + 1:]:
            for population in ("evaluated", "survivors"):
                overlaps = []
                for pair_name in sorted(runs[left]["pair_rows"]):
                    left_set = populations[population][(left, pair_name)]
                    right_set = populations[population][(right, pair_name)]
                    overlaps.append(len(left_set & right_set) / len(left_set | right_set))
                assert len(overlaps) == 28
                jaccard[(left, right, population)] = float(np.mean(overlaps))

    recurrence = {}
    for population in ("evaluated", "survivors"):
        presence = defaultdict(set)
        for key in PRIMARY_KEYS:
            for pair_name in sorted(runs[key]["pair_rows"]):
                for mechanism, expected_sign in populations[population][(key, pair_name)]:
                    presence[(pair_name, mechanism, expected_sign)].add(key)
        recurrence[population] = Counter(len(run_set) for run_set in presence.values())

    return {
        "rows": rows,
        "populations": populations,
        "frequencies": frequencies,
        "diversity": diversity,
        "jaccard": jaccard,
        "recurrence": recurrence,
        "agreement": agreement,
        "other_share": other_share,
    }


def style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#808080")
    ax.spines["bottom"].set_color("#808080")
    ax.tick_params(colors="#333333", labelsize=8)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.7, zorder=0)


def save_figure(fig, stem: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generation_figure(runs: dict[str, dict]):
    x = np.arange(10)

    def matrix(field):
        return np.array([[row[field] for row in runs[key]["generation"]] for key in PRIMARY_KEYS])

    rate = matrix("rate") * 100
    best = matrix("best_t")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.1), sharex=True)
    fig.patch.set_facecolor("white")
    (rate_l, rate_r), (best_l, best_r) = axes

    def primary_panel(ax, data, title, ylabel):
        for key, values in zip(PRIMARY_KEYS, data):
            ax.plot(x, values, color=COLORS[key], linewidth=1.15, marker="o", markersize=2.7, label=key)
        ax.fill_between(x, data.min(axis=0), data.max(axis=0), color="#315F78", alpha=0.11, linewidth=0)
        ax.plot(x, np.median(data, axis=0), color="#153E54", linewidth=2.2, label="Median")
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylabel(ylabel, fontsize=8.5)

    def contrast_panel(ax, data, comparison, title):
        ax.fill_between(x, data.min(axis=0), data.max(axis=0), color="#315F78", alpha=0.13, linewidth=0, label="B1-B3 range")
        ax.plot(x, np.median(data, axis=0), color="#153E54", linewidth=2.0, label="B1-B3 median")
        for key in ["A1", "A2", "A3"]:
            values = np.array([row[comparison] for row in runs[key]["generation"]])
            if comparison == "rate":
                values = values * 100
            ax.plot(x, values, color=COLORS[key], linewidth=1.7, marker="o", markersize=2.7, label=key)
        ax.set_title(title, fontsize=9.5, loc="left")

    primary_panel(rate_l, rate, "(a) First-pass rate, primary runs", "Correct-sign first-pass rate (%)")
    contrast_panel(rate_r, rate, "rate", "(b) First-pass rate, configuration contrasts")
    primary_panel(best_l, best, "(c) Best correct-sign $|t|$, primary runs", "Largest correct-sign $|t^{FMB}|$")
    contrast_panel(best_r, best, "best_t", "(d) Best correct-sign $|t|$, configuration contrasts")
    rate_l.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper left")
    rate_r.legend(frameon=False, fontsize=7.0, ncol=2, loc="upper left")

    for ax in (rate_l, rate_r):
        ax.set_ylim(25, 100)
    for ax in (best_l, best_r):
        ax.set_ylim(4, 9)
        ax.set_xlabel("Generation", fontsize=8.5)
    for ax in axes.flat:
        style_axes(ax)
        ax.set_xticks(x)
        ax.set_xlim(0, 9)
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save_figure(fig, "fig1_generation_dynamics")


def pair_figure(runs: dict[str, dict]):
    pair_names = sorted(runs["B1"]["pair_rows"])
    summaries = []
    for pair in pair_names:
        rates = np.array([
            runs[key]["pair_rows"][pair]["correct"] / runs[key]["pair_rows"][pair]["evaluated"] * 100
            for key in PRIMARY_KEYS
        ])
        survivors = np.array([runs[key]["pair_rows"][pair]["survivors"] for key in PRIMARY_KEYS])
        summaries.append((pair, rates, survivors))
    summaries.sort(key=lambda row: (np.median(row[1]), np.mean(row[1])))

    y = np.arange(len(summaries))
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 7.75), gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.patch.set_facecolor("white")
    rate_ax, survivor_ax = axes

    rate_low = [row[1].min() for row in summaries]
    rate_high = [row[1].max() for row in summaries]
    rate_med = [np.median(row[1]) for row in summaries]
    rate_ax.hlines(y, rate_low, rate_high, color="#AFC2CD", linewidth=2.0, zorder=1)
    rate_ax.scatter(rate_med, y, color="#153E54", s=18, zorder=2)
    rate_ax.set_yticks(y, [display_pair(row[0]) for row in summaries], fontsize=6.7)
    rate_ax.set_xlabel("Correct-sign rate (%)", fontsize=8.5)
    rate_ax.set_title("(a) First-pass yield", fontsize=9.5, loc="left")
    rate_ax.set_xlim(15, 100)

    surv_low = [row[2].min() for row in summaries]
    surv_high = [row[2].max() for row in summaries]
    surv_med = [np.median(row[2]) for row in summaries]
    survivor_ax.hlines(y, surv_low, surv_high, color="#D6AE70", linewidth=2.0, zorder=1)
    survivor_ax.scatter(surv_med, y, color="#9A5D18", s=18, zorder=2)
    survivor_ax.set_yticks(y, [])
    survivor_ax.set_xlabel("Independent survivors", fontsize=8.5)
    survivor_ax.set_title("(b) Horse-race yield", fontsize=9.5, loc="left")
    survivor_ax.set_xlim(0.5, max(surv_high) + 0.5)
    survivor_ax.set_xticks(range(1, max(surv_high) + 1))

    for ax in axes:
        style_axes(ax)
        ax.set_ylim(-0.7, len(summaries) - 0.3)
    fig.text(0.51, 0.012, "Dots show B1-B3 medians; lines show the full cross-run range.", ha="center", fontsize=7.2, color="#555555")
    fig.tight_layout(rect=[0, 0.025, 1, 1], w_pad=1.0)
    save_figure(fig, "fig2_pair_frontier")


def architecture_figure(runs: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(6.9, 4.05))
    fig.patch.set_facecolor("white")
    style_axes(ax)
    base_x = []
    base_y = []
    for key in PRIMARY_KEYS:
        totals = runs[key]["totals"]
        base_x.append(totals["significant_correct_sign"] / totals["evaluated_tests"] * 100)
        base_y.append(runs[key]["n_survivors"] / totals["evaluated_tests"] * 100)
    ax.fill_betweenx(
        [min(base_y), max(base_y)], min(base_x), max(base_x),
        color="#315F78", alpha=0.10, linewidth=0, zorder=0,
    )

    offsets = {
        "B1": (-18, -13), "B2": (-2, 8), "B3": (6, 7),
        "A1": (-28, 8), "A2": (-18, 8), "A3": (6, 8),
    }
    for key, _, _, _ in RUNS:
        totals = runs[key]["totals"]
        x = totals["significant_correct_sign"] / totals["evaluated_tests"] * 100
        y = runs[key]["n_survivors"] / totals["evaluated_tests"] * 100
        marker = "o" if key in PRIMARY_KEYS else "s"
        ax.scatter(x, y, s=46, marker=marker, color=COLORS[key], edgecolor="white", linewidth=0.7, zorder=3)
        dx, dy = offsets[key]
        ax.annotate(key, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color="#222222")

    ax.text(np.mean(base_x), min(base_y) + 0.04, "Primary-run range", ha="center", va="bottom", fontsize=7.3, color="#315F78")
    ax.set_xlabel("Correct-sign first-pass rate (%)", fontsize=9)
    ax.set_ylabel("Horse-race survivors / evaluated proposals (%)", fontsize=9)
    ax.set_xlim(37, 79)
    ax.set_ylim(3.55, 5.45)
    ax.set_title("Search yield and independent discovery yield", fontsize=10, loc="left")
    fig.tight_layout()
    save_figure(fig, "fig3_architecture_contrasts")


def discovery_funnel_table(runs: dict[str, dict]) -> str:
    rows = []
    for key, _, _, _ in RUNS:
        run = runs[key]
        t = run["totals"]
        hit = t["significant_correct_sign"] / t["evaluated_tests"] * 100
        rows.append(
            f"{key} & {t['considered_proposals']:,} & {t['validation_rejections']:,} & "
            f"{t['evaluated_tests']:,} & {t['significant_correct_sign']:,} & "
            f"{t['weak_signals']:,} & {t['wrong_sign']:,} & {t['evaluation_errors']:,} & "
            f"{hit:.1f} & {run['n_survivors']:,} \\\\"
        )
    return r"""
% ----------------------------------------------------------------------
% Table: discovery funnel
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Discovery Funnel Across Independent Runs}
\label{tab:discovery_funnel}
\scriptsize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3.3pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
Run & Consid. & Reject. & Eval. & Correct & Weak & Wrong & Error & Hit (\%) & Survivors \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\scriptsize
``Considered'' counts parsed final proposals; ``Reject.'' denotes validation
rejections before empirical evaluation. Correct, weak, wrong-sign, and error
outcomes partition the evaluated population. The hit rate is correct-sign
first passes divided by evaluated proposals. Survivors are the final outputs of
the within-run, within-pair horse races. B1--B3 are independent primary runs;
A1 is the single-agent comparison, and A2--A3 replace the proposer model while
retaining critique. Runs are not pooled.
\end{flushleft}
\end{table}
"""


def repeatability_table(runs: dict[str, dict]) -> str:
    rows = []
    unions = []
    for i, left in enumerate(PRIMARY_KEYS):
        for right in PRIMARY_KEYS[i + 1:]:
            pairs = sorted(runs[left]["pair_rows"])
            left_rates = [runs[left]["pair_rows"][pair]["correct"] / runs[left]["pair_rows"][pair]["evaluated"] for pair in pairs]
            right_rates = [runs[right]["pair_rows"][pair]["correct"] / runs[right]["pair_rows"][pair]["evaluated"] for pair in pairs]
            left_surv = [runs[left]["pair_rows"][pair]["survivors"] for pair in pairs]
            right_surv = [runs[right]["pair_rows"][pair]["survivors"] for pair in pairs]
            evaluated_overlap = len(runs[left]["evaluated_expressions"] & runs[right]["evaluated_expressions"])
            survivor_overlap = len(runs[left]["survivor_expressions"] & runs[right]["survivor_expressions"])
            varset_eval = len(runs[left]["evaluated_varsets"] & runs[right]["evaluated_varsets"])
            varset_surv = len(runs[left]["survivor_varsets"] & runs[right]["survivor_varsets"])
            unions.append((len(runs[left]["evaluated_varsets"] | runs[right]["evaluated_varsets"]),
                           len(runs[left]["survivor_varsets"] | runs[right]["survivor_varsets"])))
            rows.append(
                f"{left}--{right} & {spearman(left_rates, right_rates):.3f} & {spearman(left_surv, right_surv):.3f} & "
                f"{evaluated_overlap} & {survivor_overlap} & {varset_eval} & {varset_surv} \\\\"
            )
    n_varsets = {key: (len(runs[key]["evaluated_varsets"]), len(runs[key]["survivor_varsets"])) for key in PRIMARY_KEYS}
    per_run = "; ".join(f"{key}: {e:,} evaluated, {s} survivor" for key, (e, s) in n_varsets.items())
    return r"""
% ----------------------------------------------------------------------
% Table: cross-run repeatability
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Cross-Run Repeatability of the Primary Search}
\label{tab:baseline_repeatability}
\footnotesize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{Task-level rank correlation} & \multicolumn{2}{c}{Shared exact formulas} & \multicolumn{2}{c}{Shared variable sets} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Run pair & Hit rate & Survivors & Evaluated & Survivors & Evaluated & Survivors \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize
Rank correlations are Spearman's $\rho$ across the 28 tasks, computed separately
for task-level correct-sign hit rates and for task-level horse-race survivor
counts. Shared exact formulas require equality after deterministic expression
normalization. Shared variable sets require only that two formulas draw on the
same set of accounting variables, ignoring operators and structure; this is a
coarse structural proxy that sits between literal recurrence and the
mechanism-level coding reported in Table~\ref{tab:mechanism_recurrence}.
Distinct variable sets per run: """ + per_run + r""".
\end{flushleft}
\end{table}
"""


def select_top_survivors(runs: dict[str, dict], per_run: int = 3) -> list[tuple]:
    selected = []
    for key in PRIMARY_KEYS:
        candidates = []
        for item in runs[key]["survivor_records"]:
            cond_t = oriented(item.get("stepwise_t"), item["expected_sign"])
            alpha_t = oriented(item.get("ls_talpha"), item["expected_sign"])
            sharpe = oriented(item.get("ls_sharpe"), item["expected_sign"])
            uni_t = oriented(item.get("fmb_tstat"), item["expected_sign"])
            if None in {cond_t, alpha_t, sharpe, uni_t} or alpha_t <= 0:
                continue
            candidates.append((sharpe, alpha_t, cond_t, uni_t, item))
        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        for row in candidates[:per_run]:
            selected.append((key,) + row)
    return selected


def top_survivor_table(runs: dict[str, dict]) -> str:
    rows = []
    for key, sharpe, alpha_t, cond_t, uni_t, item in select_top_survivors(runs):
        rows.append(
            f"{key} & {tex(readable_signal(item['name']))} & {tex(display_pair(item['pair_name']))} & "
            f"{item['generation']} & {uni_t:.2f} & {cond_t:.2f} & {alpha_t:.2f} & {sharpe:.2f} \\\\"
        )
    return r"""
% ----------------------------------------------------------------------
% Table: selected primary-run survivors
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{High-Sharpe Independent Discoveries in the Primary Runs}
\label{tab:top_survivors}
\scriptsize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{2.2cm}rrrrr@{}}
\toprule
Run & Signal & Theme pair & Gen. & Univariate $t$ & Conditional $t$ & Alpha $t$ & Sharpe \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{flushleft}
\scriptsize
For each primary run, the table reports the three final horse-race survivors
with the largest sign-oriented annualized Sharpe ratios among survivors whose
Fama--French five-factor alpha has the predicted sign. Univariate $t$ is the
discovery-stage Fama--MacBeth statistic; conditional $t$ is the statistic from
the final within-pair stepwise regression. All statistics are sign-oriented
using the direction declared before evaluation. Appendix
Table~\ref{tab:top_survivor_expressions} lists the expressions. This ranking is
descriptive and does not replace search-universe inference.
\end{flushleft}
\end{table}
"""


def top_survivor_expressions_table(runs: dict[str, dict]) -> str:
    rows = []
    for key, sharpe, alpha_t, cond_t, uni_t, item in select_top_survivors(runs):
        expr = tex(str(item["expression"])).replace(", ", ",\\allowbreak{} ").replace("(", "(\\allowbreak{}")
        rows.append(
            f"{key} & {tex(readable_signal(item['name']))} & {item['expected_sign']} & \\texttt{{{expr}}} \\\\"
        )
    return r"""
% ----------------------------------------------------------------------
% Table: expressions of the selected survivors (appendix)
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Expressions of the Selected Primary-Run Survivors}
\label{tab:top_survivor_expressions}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}p{3.2cm}l>{\raggedright\arraybackslash}X@{}}
\toprule
Run & Signal & Sign & Expression \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{flushleft}
\scriptsize
Expressions are reproduced exactly as proposed and evaluated. ``Sign'' is the
direction declared by the proposer before evaluation.
\end{flushleft}
\end{table}
"""


def theme_marginal_table(runs: dict[str, dict]) -> str:
    rows = []
    for theme_key, theme_label in THEME_KEYS.items():
        rates = []
        survivors = []
        for key in PRIMARY_KEYS:
            correct = evaluated = surv = 0
            for pair, row in runs[key]["pair_rows"].items():
                if theme_key in pair.split("_", 1):
                    correct += row["correct"]
                    evaluated += row["evaluated"]
                    surv += row["survivors"]
            rates.append(correct / evaluated * 100)
            survivors.append(surv / 7)
        rows.append((np.median(rates), theme_label, rates, survivors))
    rows.sort(reverse=True)
    body = "\n".join(
        f"{label} & {r[0]:.1f} & {r[1]:.1f} & {r[2]:.1f} & {np.median(r):.1f} [{min(r):.1f}, {max(r):.1f}] & "
        f"{np.median(s):.2f} [{min(s):.2f}, {max(s):.2f}] \\\\"
        for _, label, r, s in rows
    )
    return r"""
% ----------------------------------------------------------------------
% Table: theme-level marginal yield (appendix)
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Theme-Level Marginal Discovery Yield in the Primary Runs}
\label{tab:theme_marginals}
\footnotesize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Theme & B1 hit & B2 hit & B3 hit & Median [range] & Survivors per task, median [range] \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize
Each theme participates in seven tasks. The hit rate pools the evaluated
proposals of those seven tasks within a run; survivors per task average the
final horse-race survivors over the same seven tasks. Runs are kept separate.
\end{flushleft}
\end{table}
"""


def mechanism_table(mechanisms: dict | None) -> str:
    """Mechanism-level recurrence across B1--B3.

    Reads a blinded mechanism coding file if present; otherwise emits a
    placeholder with the intended layout so that the text can reference it.
    Expected columns in mechanism_codes.csv: run, pair_name, signal_name,
    expression, primary_mechanism, expected_sign.
    """
    if mechanisms is None:
        return r"""
% ----------------------------------------------------------------------
% Table: mechanism-level recurrence (PLACEHOLDER until mechanism_codes.csv exists)
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Mechanism-Level Recurrence Across the Primary Runs}
\label{tab:mechanism_recurrence}
\footnotesize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
Population & Pair-mechanisms in 3/3 runs & In exactly 2/3 runs & In 1/3 runs \\
\midrule
Evaluated hypotheses & \multicolumn{3}{c}{\textit{pending blinded mechanism coding}} \\
Horse-race survivors & \multicolumn{3}{c}{\textit{pending blinded mechanism coding}} \\
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize
A pair-mechanism is one (task, primary mechanism, predicted sign) label
assigned under a codebook frozen before any post-search statistic was viewed.
Coding uses the hypothesis, expression, sign, and task only. This table is
generated from \texttt{paper/generated/mechanism\_codes.csv}; the file does
not yet exist, so the layout is shown without values.
\end{flushleft}
\end{table}
"""
    lines = []
    for population, title in [("evaluated", "Evaluated hypotheses"), ("survivors", "Horse-race survivors")]:
        counts = mechanisms["recurrence"][population]
        lines.append(f"{title} & {counts[3]} & {counts[2]} & {counts[1]} \\\\")
    overlap_lines = []
    for left_index, left in enumerate(PRIMARY_KEYS):
        for right in PRIMARY_KEYS[left_index + 1:]:
            candidate = mechanisms["jaccard"][(left, right, "evaluated")]
            survivor = mechanisms["jaccard"][(left, right, "survivors")]
            overlap_lines.append(f"{left}--{right} & {candidate:.3f} & {survivor:.3f} \\\\")
    agreement = mechanisms["agreement"]
    return r"""
% ----------------------------------------------------------------------
% Table: mechanism-level recurrence
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Mechanism-Level Recurrence Across the Primary Runs}
\label{tab:mechanism_recurrence}
\footnotesize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{6pt}
\textit{Panel A: Pair-mechanism-sign recurrence}\\[3pt]
\begin{tabular}{@{}lrrr@{}}
\toprule
Population & Pair-mechanisms in 3/3 runs & In exactly 2/3 runs & In 1/3 runs \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\vspace{7pt}

\textit{Panel B: Macro-averaged pair-level Jaccard overlap}\\[3pt]
\begin{tabular}{@{}lrr@{}}
\toprule
Run pair & Evaluated hypotheses & Horse-race survivors \\
\midrule
""" + "\n".join(overlap_lines) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize
A pair-mechanism is one (task, primary mechanism, predicted sign) label
assigned under a codebook frozen before any post-search statistic was viewed.
Coding uses the hypothesis, expression, sign, and task only. Panel A counts
distinct pair-mechanism-sign labels by how many primary runs contain at least
one hypothesis carrying that label. Panel B computes Jaccard overlap of the
mechanism-sign sets separately within each task and then averages equally over
the 28 tasks. The two independent coding passes agreed on
""" + f"{agreement['exact_agreement'] * 100:.1f}" + r"""\% of records
(Cohen's $\kappa=""" + f"{agreement['cohen_kappa']:.3f}" + r"""$); all
disagreements were independently adjudicated. The residual Other label accounts
for """ + f"{mechanisms['other_share'] * 100:.2f}" + r"""\% of records.
\end{flushleft}
\end{table}
"""


def mechanism_diversity_table(mechanisms: dict | None) -> str:
    """Run-level summary of economic-mechanism breadth and concentration."""
    if mechanisms is None:
        return ""
    pair_names = sorted(pair for run, pair in mechanisms["diversity"] if run == "B1")
    assert len(pair_names) == 28
    rows = []
    for key, *_ in RUNS:
        values = [mechanisms["diversity"][(key, pair)] for pair in pair_names]
        counts = [value["count"] for value in values]
        hhis = [value["hhi"] for value in values]
        rows.append(
            f"{key} & {np.mean(counts):.2f} & {np.median(counts):.0f} [{min(counts)}, {max(counts)}] & "
            f"{np.mean(hhis):.3f} & {np.median(hhis):.3f} [{min(hhis):.3f}, {max(hhis):.3f}] \\\\")
    return r"""
% ----------------------------------------------------------------------
% Table: mechanism diversity by run
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Mechanism Breadth and Concentration Across Tasks}
\label{tab:mechanism_diversity}
\footnotesize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{5.5pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Run & Mean count & Median count [range] & Mean HHI & Median HHI [range] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize
Statistics are computed separately for each of the 28 tasks from all evaluated
hypotheses in a run. Count is the number of distinct substantive mechanism
labels used in the task. HHI is $\sum_m s_m^2$, where $s_m$ is a mechanism's
share among non-Other records in that run and task; lower values denote a more
even distribution across mechanisms. The residual Other label is excluded
from both measures. B1--B3 are independent primary runs, A1 is the single-agent
comparison, and A2--A3 replace the proposer model while retaining critique.
\end{flushleft}
\end{table}
"""


def mechanism_diversity_pair_table(mechanisms: dict | None) -> str:
    """Appendix detail for the pair-level A1--A3 diversity diagnostics."""
    if mechanisms is None:
        return ""
    pair_names = sorted(pair for run, pair in mechanisms["diversity"] if run == "B1")
    assert len(pair_names) == 28
    rows = []
    for pair in pair_names:
        base = [mechanisms["diversity"][(key, pair)] for key in PRIMARY_KEYS]
        base_counts = [value["count"] for value in base]
        base_hhis = [value["hhi"] for value in base]
        alternatives = [mechanisms["diversity"][(key, pair)] for key in ("A1", "A2", "A3")]
        rows.append(
            f"{tex(display_pair(pair))} & "
            f"{np.median(base_counts):.0f} [{min(base_counts)}, {max(base_counts)}] & "
            + " & ".join(str(value["count"]) for value in alternatives)
            + f" & {np.median(base_hhis):.3f} [{min(base_hhis):.3f}, {max(base_hhis):.3f}] & "
            + " & ".join(f"{value['hhi']:.3f}" for value in alternatives)
            + r" \\"
        )
    return r"""
% ----------------------------------------------------------------------
% Table: pair-level mechanism diversity (appendix)
% ----------------------------------------------------------------------

\clearpage
\begingroup
\scriptsize
\renewcommand{\arraystretch}{1.08}
\setlength{\tabcolsep}{3.2pt}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{3.1cm}cccccccc@{}}
\caption{Pair-Level Mechanism Breadth and Concentration}
\label{tab:mechanism_diversity_pairs} \\
\toprule
& \multicolumn{4}{c}{Distinct mechanism count} & \multicolumn{4}{c}{HHI} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Theme pair & B median [range] & A1 & A2 & A3 & B median [range] & A1 & A2 & A3 \\
\midrule
\endfirsthead
\multicolumn{9}{l}{\textit{Table~\ref{tab:mechanism_diversity_pairs} continued}} \\[4pt]
\toprule
& \multicolumn{4}{c}{Distinct mechanism count} & \multicolumn{4}{c}{HHI} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Theme pair & B median [range] & A1 & A2 & A3 & B median [range] & A1 & A2 & A3 \\
\midrule
\endhead
\midrule
\multicolumn{9}{r}{\textit{Continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(rows) + r"""
\end{longtable}
\begin{flushleft}
\scriptsize
B summarizes the median and full range across B1--B3 without pooling records.
A1--A3 are the three single-run configuration comparisons. Counts and HHI are
computed within a run and task from evaluated hypotheses, excluding the
residual Other label; lower HHI indicates less concentration. A1 is the direct
no-critique contrast because it retains the primary proposer. A2--A3 instead
change proposer identity and are descriptive model contrasts.
\end{flushleft}
\endgroup
"""


def pair_results_table(runs: dict[str, dict]) -> str:
    pair_names = sorted(runs["B1"]["pair_rows"])
    rows = []
    for pair in pair_names:
        rates = [runs[key]["pair_rows"][pair]["correct"] / runs[key]["pair_rows"][pair]["evaluated"] * 100 for key in PRIMARY_KEYS]
        survivors = [runs[key]["pair_rows"][pair]["survivors"] for key in PRIMARY_KEYS]
        rows.append(
            f"{tex(display_pair(pair))} & {rates[0]:.1f} & {rates[1]:.1f} & {rates[2]:.1f} & "
            f"{np.median(rates):.1f} [{min(rates):.1f}, {max(rates):.1f}] & "
            f"{np.median(survivors):.0f} [{min(survivors)}, {max(survivors)}] \\\\"
        )
    return r"""
% ----------------------------------------------------------------------
% Table: task-level primary results (appendix)
% ----------------------------------------------------------------------

\clearpage
\begingroup
\scriptsize
\renewcommand{\arraystretch}{1.08}
\setlength{\tabcolsep}{4pt}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{4.0cm}rrrrr@{}}
\caption{Primary Discovery Yield by Theme Pair}
\label{tab:baseline_pair_results} \\
\toprule
Theme pair & B1 hit & B2 hit & B3 hit & Median [range] & Survivors median [range] \\
\midrule
\endfirsthead
\multicolumn{6}{l}{\textit{Table~\ref{tab:baseline_pair_results} continued}} \\[4pt]
\toprule
Theme pair & B1 hit & B2 hit & B3 hit & Median [range] & Survivors median [range] \\
\midrule
\endhead
\midrule
\multicolumn{6}{r}{\textit{Continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(rows) + r"""
\end{longtable}
\begin{flushleft}
\scriptsize
Hit rates are percentages of evaluated proposals that pass the declared-sign
Fama--MacBeth screen. Brackets give the full B1--B3 range; each primary run is
kept separate. Survivor counts come from the final within-pair horse race.
\end{flushleft}
\endgroup
"""


def process_table(runs: dict[str, dict]) -> str:
    rows = []
    for key, _, _, _ in RUNS:
        run = runs[key]
        proposer = run["calls"].get("proposer", 0)
        critic = run["calls"].get("critic", 0)
        revise = max(proposer - 280, 0)
        flag_rate = "---" if critic == 0 else f"{run['critic_flags'] / 1680 * 100:.1f}"
        rows.append(
            f"{key} & {run['n_calls']:,} & {proposer:,} & {critic:,} & {revise:,} & "
            f"{run['output_tokens'] / 1_000_000:.2f} & {flag_rate} \\\\"
        )
    return r"""
% ----------------------------------------------------------------------
% Table: model-call accounting (appendix)
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Model-Call Accounting by Run}
\label{tab:process_accounting}
\footnotesize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Run & Calls & Proposer calls & Critic calls & Revision calls & Output tokens (m) & Flagged (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize
Counts come from recorded model responses. Proposer calls include one initial
call per pair-generation and any revision calls; revision calls are therefore
proposer calls minus 280. The flagged share divides critic flags by the 1,680
requested proposal slots. Output-token counts are recorded provider usage and
are reported as process quantities rather than equal-compute controls.
\end{flushleft}
\end{table}
"""


def write_tables(runs: dict[str, dict], mechanisms: dict | None):
    path = PAPER_DIR / "tables.tex"
    header = """% ======================================================================
%  paper/tables.tex
%  Auto-generated by the section-prefixed scripts in paper/scripts/.
%  Do not edit generated table blocks by hand.
% ======================================================================
"""
    begin = "% BEGIN SECTION5 TABLES"
    end = "% END SECTION5 TABLES"
    block = begin + "\n" + "".join([
        discovery_funnel_table(runs),
        repeatability_table(runs),
        mechanism_table(mechanisms),
        mechanism_diversity_table(mechanisms),
        top_survivor_table(runs),
        pair_results_table(runs),
        theme_marginal_table(runs),
        mechanism_diversity_pair_table(mechanisms),
        top_survivor_expressions_table(runs),
        process_table(runs),
    ]) + "\n" + end
    if path.exists() and "% BEGIN SECTION" in path.read_text():
        current = path.read_text()
        if begin in current and end in current:
            prefix, rest = current.split(begin, 1)
            _, suffix = rest.split(end, 1)
            current = prefix.rstrip() + "\n\n" + block + suffix
        else:
            current = current.rstrip() + "\n\n" + block + "\n"
    else:
        current = header + "\n" + block + "\n"
    path.write_text(current)


def validate_and_report(runs: dict[str, dict], mechanisms: dict | None):
    for key, _, _, _ in RUNS:
        run = runs[key]
        totals = run["totals"]
        assert sum(row["evaluated"] for row in run["generation"]) == totals["evaluated_tests"]
        assert sum(row["correct"] for row in run["generation"]) == totals["significant_correct_sign"]
    base_hits = [runs[key]["totals"]["significant_correct_sign"] / runs[key]["totals"]["evaluated_tests"] * 100 for key in PRIMARY_KEYS]
    base_survivors = [runs[key]["n_survivors"] for key in PRIMARY_KEYS]
    print(f"Primary hit rate: median {np.median(base_hits):.2f}%, range {min(base_hits):.2f}-{max(base_hits):.2f}%")
    print(f"Primary survivors: median {np.median(base_survivors):.0f}, range {min(base_survivors)}-{max(base_survivors)}")
    for key, _, _, _ in RUNS:
        run = runs[key]
        totals = run["totals"]
        print(
            f"{key}: evaluated={totals['evaluated_tests']}, correct={totals['significant_correct_sign']}, "
            f"survivors={run['n_survivors']}, calls={run['n_calls']}"
        )
        print(f"  rejections={run['rejections']} gen0={run['rejections_gen0']}")
        print("  evaluated by gen:", [row["evaluated"] for row in run["generation"]])
        print("  best |t| by gen :", [round(row["best_t"], 2) for row in run["generation"]])
        print("  rate by gen     :", [round(row["rate"] * 100, 1) for row in run["generation"]])
    if mechanisms is not None:
        agreement = mechanisms["agreement"]
        print(
            f"Mechanism coding: exact agreement={agreement['exact_agreement']:.4f}, "
            f"kappa={agreement['cohen_kappa']:.4f}, Other={mechanisms['other_share']:.2%}"
        )
        for left_index, left in enumerate(PRIMARY_KEYS):
            for right in PRIMARY_KEYS[left_index + 1:]:
                print(
                    f"  {left}-{right} macro Jaccard: "
                    f"evaluated={mechanisms['jaccard'][(left, right, 'evaluated')]:.4f}, "
                    f"survivors={mechanisms['jaccard'][(left, right, 'survivors')]:.4f}"
                )
        pair_names = sorted(pair for run, pair in mechanisms["diversity"] if run == "B1")
        for key, *_ in RUNS:
            values = [mechanisms["diversity"][(key, pair)] for pair in pair_names]
            print(
                f"  {key} mechanism diversity: mean count={np.mean([v['count'] for v in values]):.3f}, "
                f"mean HHI={np.mean([v['hhi'] for v in values]):.4f}"
            )


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlecolor": "#222222",
        "axes.labelcolor": "#222222",
        "savefig.facecolor": "white",
    })
    runs = {
        key: collect_run(key, label, dirname, primary)
        for key, label, dirname, primary in RUNS
    }
    mechanisms = load_mechanism_analysis(runs)
    generation_figure(runs)
    pair_figure(runs)
    architecture_figure(runs)
    write_tables(runs, mechanisms)
    validate_and_report(runs, mechanisms)


if __name__ == "__main__":
    main()
