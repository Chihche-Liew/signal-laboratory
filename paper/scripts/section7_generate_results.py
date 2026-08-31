#!/usr/bin/env python3
"""Generate Section 7 critique-taxonomy figures and LaTeX tables.

The input is the adjudicated, outcome-blind critique coding file produced by
``section7_classify_critiques.py``.  This script never reads discovery-stage
evaluation results or post-search survivor artifacts.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "signallab-paper-mpl"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
GENERATED_DIR = PAPER_DIR / "generated"
FIGURE_DIR = PAPER_DIR / "figures"
CODES_PATH = GENERATED_DIR / "critique_codes.csv"
METRICS_PATH = GENERATED_DIR / "critique_coding_metrics.json"
SUMMARY_PATH = GENERATED_DIR / "section7_critique_summary.json"
TABLES_PATH = PAPER_DIR / "tables.tex"

RUNS = ["B1", "B2", "B3", "A2", "A3"]
PRIMARY_ORDER = [
    "novelty_redundancy",
    "economic_logic",
    "measurement_construct",
    "identification_attribution",
    "functional_form",
    "data_feasibility",
    "archive_feedback",
]
PRIMARY_LABELS = {
    "novelty_redundancy": "Novelty / redundancy",
    "economic_logic": "Economic logic",
    "measurement_construct": "Measurement / construct",
    "identification_attribution": "Identification / attribution",
    "functional_form": "Functional form",
    "data_feasibility": "Data feasibility",
    "archive_feedback": "Archive feedback",
}
PRIMARY_SHORT_LABELS = {
    "novelty_redundancy": "Novelty /\nredundancy",
    "economic_logic": "Economic\nlogic",
    "measurement_construct": "Measurement /\nconstruct",
    "identification_attribution": "Identification /\nattribution",
    "functional_form": "Functional\nform",
    "data_feasibility": "Data\nfeasibility",
    "archive_feedback": "Archive\nfeedback",
}
NOVELTY_ORDER = [
    "published_rediscovery",
    "archive_duplicate_or_cosmetic_variant",
    "limited_incremental_extension",
    "substantive_new_condition_or_combination",
]
NOVELTY_LABELS = {
    "published_rediscovery": "Published rediscovery",
    "archive_duplicate_or_cosmetic_variant": "Archive duplicate / cosmetic variant",
    "limited_incremental_extension": "Limited incremental extension",
    "substantive_new_condition_or_combination": "Substantive new condition / combination",
    "not_discussed": "Novelty not discussed",
}
NOVELTY_FIGURE_LABELS = {
    "published_rediscovery": "Published\nrediscovery",
    "archive_duplicate_or_cosmetic_variant": "Archive duplicate /\ncosmetic variant",
    "limited_incremental_extension": "Limited incremental\nextension",
    "substantive_new_condition_or_combination": "Substantive new\ncondition / combination",
}


def read_rows() -> list[dict]:
    with CODES_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8398
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert {row["run"] for row in rows} == set(RUNS)
    assert all(row["primary_target"] in {"none", *PRIMARY_ORDER} for row in rows)
    assert all(row["novelty_assessment"] in {"not_discussed", *NOVELTY_ORDER} for row in rows)
    return rows


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURE_DIR / f"{stem}.{suffix}",
            dpi=240,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


def taxonomy_figure(rows: list[dict]) -> None:
    critique_rows = [row for row in rows if row["primary_target"] != "none"]
    mentions = []
    for row in critique_rows:
        labels = [row["primary_target"], *[label for label in row["secondary_targets"].split("|") if label]]
        assert len(labels) == len(set(labels))
        mentions.extend((label, row["verdict"]) for label in labels)
    flag_counts = Counter(label for label, verdict in mentions if verdict == "flag")
    caveat_counts = Counter(label for label, verdict in mentions if verdict == "accept")
    counts = Counter(label for label, _ in mentions)
    ordered = sorted(PRIMARY_ORDER, key=lambda label: counts[label])

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), gridspec_kw={"width_ratios": [1.0, 1.32]})
    fig.patch.set_facecolor("white")
    color = "#355C7D"
    axes[0].barh(
        [PRIMARY_LABELS[label] for label in ordered],
        [flag_counts[label] for label in ordered],
        color=color,
        label="Flagged",
    )
    axes[0].barh(
        [PRIMARY_LABELS[label] for label in ordered],
        [caveat_counts[label] for label in ordered],
        left=[flag_counts[label] for label in ordered],
        color="#A9BDC9",
        label="Accepted with caveat",
    )
    axes[0].set_xlabel("Number of notes (non-exclusive)", fontsize=8)
    axes[0].set_title("(a) Issues raised by the critic", loc="left", fontsize=8.8)
    axes[0].grid(axis="x", color="#dddddd", linewidth=0.7)
    axes[0].set_axisbelow(True)
    for y, label in enumerate(ordered):
        axes[0].text(counts[label] + max(counts.values()) * 0.015, y, f"{counts[label]:,}", va="center", fontsize=6.7)
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower right")

    matrix = np.array(
        [
            [
                pct(
                    sum(row["run"] == run and row["primary_target"] == label for row in critique_rows),
                    sum(row["run"] == run for row in critique_rows),
                )
                for label in PRIMARY_ORDER
            ]
            for run in RUNS
        ]
    )
    image = axes[1].imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=max(25, float(matrix.max())))
    axes[1].set_xticks(range(len(PRIMARY_ORDER)), [PRIMARY_SHORT_LABELS[label] for label in PRIMARY_ORDER], rotation=35, ha="right")
    axes[1].set_yticks(range(len(RUNS)), RUNS)
    axes[1].set_title("(b) Composition within run (percent)", loc="left", fontsize=8.8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[1].text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=6.2, color="white" if matrix[i, j] > matrix.max() * 0.55 else "#222222")
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.035, pad=0.03)
    cbar.set_label("Percent")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6.6, colors="#3F4E53", length=2.5)
    fig.tight_layout()
    save_figure(fig, "fig7_critique_taxonomy")


def novelty_figure(rows: list[dict]) -> None:
    discussed = [row for row in rows if row["novelty_assessment"] != "not_discussed"]
    counts = Counter(row["novelty_assessment"] for row in discussed)
    flag_share = {
        label: pct(
            sum(row["novelty_assessment"] == label and row["verdict"] == "flag" for row in discussed),
            counts[label],
        )
        for label in NOVELTY_ORDER
    }
    short_labels = [
        "Published rediscovery",
        "Archive duplicate / cosmetic variant",
        "Limited incremental extension",
        "Substantive new condition / combination",
    ]
    y = np.arange(len(short_labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharey=True, gridspec_kw={"width_ratios": [1.05, 0.95]})
    fig.patch.set_facecolor("white")
    count_values = [counts[label] for label in NOVELTY_ORDER]
    axes[0].barh(y, count_values, color="#4C78A8")
    axes[0].set_yticks(y, short_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Number of explicit assessments", fontsize=8)
    axes[0].set_title("(a) Contemporaneous novelty assessment", loc="left", fontsize=8.8)
    axes[0].grid(axis="x", color="#dddddd", linewidth=0.7)
    axes[0].set_axisbelow(True)
    for row_index, value in enumerate(count_values):
        axes[0].text(value + max(count_values) * 0.015, row_index, f"{value:,}", va="center", fontsize=6.8)
    flag_values = [flag_share[label] for label in NOVELTY_ORDER]
    axes[1].barh(y, flag_values, color="#C44E52")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Flagged by critic (percent)", fontsize=8)
    axes[1].set_title("(b) Critic disposition within class", loc="left", fontsize=8.8)
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].tick_params(labelleft=False)
    for row_index, value in enumerate(flag_values):
        axes[1].text(min(value + 2, 97), row_index, f"{value:.0f}", va="center", fontsize=6.8)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6.5, colors="#3F4E53", length=2.5)
    fig.tight_layout()
    save_figure(fig, "fig8_critique_novelty")


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def table_block(rows: list[dict]) -> str:
    critique_rows = [row for row in rows if row["primary_target"] != "none"]
    target_counts = Counter(row["primary_target"] for row in critique_rows)
    novelty_counts = Counter(row["novelty_assessment"] for row in rows)
    discussed_n = len(rows) - novelty_counts["not_discussed"]
    target_lines = []
    for label in sorted(PRIMARY_ORDER, key=lambda item: target_counts[item], reverse=True):
        per_run = [sum(row["run"] == run and row["primary_target"] == label for row in critique_rows) for run in RUNS]
        target_lines.append(
            f"{latex_escape(PRIMARY_LABELS[label])} & {target_counts[label]:,} & {pct(target_counts[label], len(critique_rows)):.1f}"
            + "".join(f" & {value:,}" for value in per_run)
            + r" \\"
        )
    novelty_lines = []
    for label in [*NOVELTY_ORDER, "not_discussed"]:
        denominator = discussed_n if label != "not_discussed" else len(rows)
        flagged = sum(row["novelty_assessment"] == label and row["verdict"] == "flag" for row in rows)
        novelty_lines.append(
            f"{latex_escape(NOVELTY_LABELS[label])} & {novelty_counts[label]:,} & {pct(novelty_counts[label], denominator):.1f} & {pct(flagged, novelty_counts[label]):.1f}"
            + r" \\"
        )
    return r"""
% BEGIN SECTION7 TABLES

\clearpage
\begin{table}[t]
\centering
\caption{Taxonomy of Contemporaneous Critic Objections}
\label{tab:critique_taxonomy}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Dominant object & Total & Share (\%) & B1 & B2 & B3 & A2 & A3 \\
\midrule
""" + "\n".join(target_lines) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\scriptsize
The table classifies the dominant object of criticism in contemporaneous critic notes that contain a material objection. Categories are mutually exclusive. ``Archive feedback'' identifies failure to use a diagnostic already visible within the same task; simple formula similarity is coded as novelty/redundancy. No evaluation outcome for the criticized proposal, survivor status, or post-search result enters the coding corpus.
\end{flushleft}
\end{table}

\begin{table}[t]
\centering
\caption{Critic Assessments of Proposal Novelty}
\label{tab:critique_novelty}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Assessment & Count & Share (\%) & Flagged (\%) \\
\midrule
""" + "\n".join(novelty_lines) + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\scriptsize
Shares for the first four rows use only notes that make an explicit novelty comparison; the residual row reports its share of all critic notes. ``Flagged'' is the critic's contemporaneous disposition, not an empirical outcome. The comparisons are limited to published signals invoked by the critic and the visible within-task archive; they are not exhaustive literature classifications.
\end{flushleft}
\end{table}

% END SECTION7 TABLES
"""


def replace_table_block(block: str) -> None:
    text = TABLES_PATH.read_text() if TABLES_PATH.exists() else ""
    begin = "% BEGIN SECTION7 TABLES"
    end = "% END SECTION7 TABLES"
    if begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + block.strip() + text[stop:]
    else:
        text = text.rstrip() + "\n\n" + block.strip() + "\n"
    TABLES_PATH.write_text(text)


def main() -> None:
    rows = read_rows()
    metrics = json.loads(METRICS_PATH.read_text())
    taxonomy_figure(rows)
    novelty_figure(rows)
    replace_table_block(table_block(rows))

    critique_rows = [row for row in rows if row["primary_target"] != "none"]
    novelty_discussed = [row for row in rows if row["novelty_assessment"] != "not_discussed"]
    summary = {
        "n_records": len(rows),
        "n_material_critiques": len(critique_rows),
        "material_critique_share": len(critique_rows) / len(rows),
        "primary_counts": Counter(row["primary_target"] for row in rows),
        "novelty_counts": Counter(row["novelty_assessment"] for row in rows),
        "novelty_discussed_share": len(novelty_discussed) / len(rows),
        "verdict_counts": Counter(row["verdict"] for row in rows),
        "coding_metrics": metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
