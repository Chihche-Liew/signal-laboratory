#!/usr/bin/env python3
"""Generate the Section 6 testing-dimension figures.

The figures follow the cumulative populations declared in Section 6:

* factor-alphas: 207 candidates entering the three-model alpha gate;
* CZ spanning: 103 candidates entering the published-predictor gate;
* subsamples: the 11 complete-protocol signals.

Run from any working directory:

    python paper/scripts/section6_generate_figures.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "signallab-paper-mpl"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
FIGURE_DIR = PAPER_DIR / "figures"
POSTHOC_DIR = PAPER_DIR / "data" / "formal" / "posthoc"
CATALOG_PATH = PAPER_DIR / "data" / "signal_catalog.json"

RUNS = [
    ("B1", "baseline_r1"),
    ("B2", "baseline_r2"),
    ("B3", "baseline_r3"),
    ("A1", "ablation_no_critique"),
    ("A2", "ablation_gpt_5_5"),
    ("A3", "ablation_gpt_5_6_sol"),
]
COLORS = {
    "B1": "#6F91A8",
    "B2": "#315F78",
    "B3": "#9DB6C5",
    "A1": "#B24C3D",
    "A2": "#7A5AA6",
    "A3": "#D18B2C",
}
WINDOWS = [
    ("full", "Full"),
    ("pre_anomaly", "Pre-1991"),
    ("post_publication", "Post-1990"),
    ("pre_2000", "Pre-2000"),
    ("post_2000", "Post-2000"),
    ("post_2010", "Post-2010"),
    ("ex_recession", "Ex-recession"),
]


def read_json(path: Path):
    return json.loads(path.read_text())


def exact_named_row(rows: list[dict], name: str, t_stat: float) -> dict:
    named = [row for row in rows if row.get("name") == name]
    if not named:
        raise AssertionError(f"no row for {name}")
    return min(named, key=lambda row: abs(float(row["t_stat"]) - float(t_stat)))


def readable_signal(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A969A")
    ax.spines["bottom"].set_color("#8A969A")
    ax.tick_params(labelsize=7.4, colors="#3F4E53", length=3)
    ax.grid(color="#DDE3E5", linewidth=0.55, alpha=0.8, zorder=0)


def save_figure(fig, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def collect_rows() -> list[dict]:
    rows = []
    for run, dirname in RUNS:
        root = POSTHOC_DIR / dirname
        multiple = read_json(root / "multiple_testing" / "results.json")
        mt_rows = multiple["multiple_testing"]["per_signal"]
        bayes_rows = multiple["bayesian"]["signals"]
        hurdle = float(
            read_json(root / "double_bootstrap" / "results.json")["double_bootstrap"]["t_hurdle_005"]
        )
        subsample_payload = read_json(root / "subsample" / "results.json")
        subsample = {row["signal_id"]: row for row in subsample_payload["subsample"]}
        decay = {row["signal_id"]: row for row in subsample_payload["decay"]}
        spanning = {row["signal_id"]: row for row in read_json(root / "spanning" / "results.json")}

        for alpha in read_json(root / "multi_model_alpha" / "results.json"):
            signal_id = alpha["signal_id"]
            full_t = float(alpha["full_fmb_t"])
            sign = 1.0 if full_t >= 0 else -1.0
            mt = exact_named_row(mt_rows, alpha["name"], full_t)
            bayes = exact_named_row(bayes_rows, alpha["name"], full_t)
            passed = math.isfinite(float(alpha["horse_race_t"])) and abs(float(alpha["horse_race_t"])) >= 1.96
            passed = passed and float(alpha["horse_race_t"]) * full_t > 0
            passed = passed and mt.get("bhy_reject") is True
            passed = passed and float(bayes["bayes_p"]["bayes_p_020"]) < 0.05
            passed = passed and abs(full_t) >= hurdle
            passed = passed and int(subsample[signal_id]["n_robust"]) >= 4
            passed_to_decay = passed
            passed = passed and decay[signal_id]["classification"] != "decaying"
            passed_to_alpha = passed
            alpha_pass = all(
                alpha.get(f"{model}_status") == "ok"
                and alpha.get(f"{model}_direction_consistent") is True
                and alpha.get(f"{model}_survives_196") is True
                for model in ("FF5", "FF6", "q4")
            )
            passed = passed and alpha_pass
            rows.append(
                {
                    "run": run,
                    "signal_id": signal_id,
                    "name": alpha["name"],
                    "sign": sign,
                    "full_t": full_t,
                    "alpha": alpha,
                    "subsample": subsample[signal_id],
                    "decay": decay[signal_id],
                    "spanning": spanning[signal_id],
                    "passed_to_decay": passed_to_decay,
                    "passed_to_alpha": passed_to_alpha,
                    "alpha_pass": alpha_pass,
                    "passed_to_spanning": passed,
                }
            )
    assert len(rows) == 449
    assert sum(row["passed_to_decay"] for row in rows) == 325
    assert sum(row["passed_to_alpha"] for row in rows) == 207
    assert sum(row["passed_to_spanning"] for row in rows) == 103
    return rows


def factor_alpha_figure(rows: list[dict]) -> dict[str, int]:
    population = [row for row in rows if row["passed_to_alpha"]]
    models = [("FF5", "FF5"), ("FF6", "FF6"), ("q4", "$q$-factor")]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.85), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    counts = {}
    for ax, (model, label) in zip(axes, models):
        style_axes(ax)
        for run, _ in RUNS:
            subset = [
                row for row in population
                if row["run"] == run
                and row["alpha"].get(f"{model}_status") == "ok"
                and row["alpha"].get(f"{model}_talpha") is not None
                and math.isfinite(float(row["alpha"][f"{model}_talpha"]))
            ]
            x = [abs(row["full_t"]) for row in subset]
            y = [row["sign"] * float(row["alpha"][f"{model}_talpha"]) for row in subset]
            ax.scatter(
                x, y, s=16, color=COLORS[run], alpha=0.70,
                edgecolor="white", linewidth=0.25, label=run, zorder=2,
            )
        count = sum(
            row["alpha"].get(f"{model}_status") == "ok"
            and row["alpha"].get(f"{model}_talpha") is not None
            and row["sign"] * float(row["alpha"][f"{model}_talpha"]) > 1.96
            for row in population
        )
        counts[model] = count
        ax.axhline(1.96, color="#9B3F34", linestyle="--", linewidth=0.9, zorder=1)
        ax.axhline(0, color="#738186", linewidth=0.6, zorder=1)
        ax.plot([0, 9], [0, 9], color="#89969A", linestyle=":", linewidth=0.8, zorder=1)
        ax.set_xlim(1.9, 8.7)
        ax.set_ylim(-5.8, 8.7)
        panel = {"FF5": "(a)", "FF6": "(b)", "q4": "(c)"}[model]
        ax.set_title(f"{panel} {label}: {count}/207 pass", fontsize=8.8, loc="left")
        ax.set_xlabel("Sign-oriented FMB $t$", fontsize=8.2)
    axes[0].set_ylabel("Sign-oriented alpha $t$", fontsize=8.2)
    axes[-1].legend(frameon=False, fontsize=6.7, ncol=2, loc="lower right")
    fig.text(
        0.5, 0.005,
        "Population: 207 candidates entering the all-model-alpha gate; the dotted line is equality.",
        ha="center", fontsize=7.0, color="#536166",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1], w_pad=1.0)
    save_figure(fig, "fig4_multi_model")
    return counts


def spanning_figure(rows: list[dict]) -> tuple[int, int]:
    population = [row for row in rows if row["passed_to_spanning"]]
    estimable = [row for row in population if row["spanning"].get("status") == "ok"]
    unavailable = len(population) - len(estimable)
    passed = [
        row for row in estimable
        if row["spanning"].get("direction_consistent") is True
        and row["sign"] * float(row["spanning"]["fmb_t_conditional"]) > 1.96
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.05))
    fig.patch.set_facecolor("white")
    style_axes(ax)
    for row in estimable:
        x = row["sign"] * float(row["spanning"]["fmb_t_univariate"])
        y = row["sign"] * float(row["spanning"]["fmb_t_conditional"])
        is_pass = row in passed
        ax.scatter(
            x, y, s=35 if is_pass else 19,
            marker="D" if is_pass else "o",
            color="#16766F" if is_pass else COLORS[row["run"]],
            alpha=0.92 if is_pass else 0.58,
            edgecolor="white", linewidth=0.4, zorder=3 if is_pass else 2,
        )
    ax.axhline(1.96, color="#9B3F34", linestyle="--", linewidth=0.95)
    ax.axhline(0, color="#738186", linewidth=0.6)
    ax.plot([0, 9], [0, 9], color="#89969A", linestyle=":", linewidth=0.85)
    ax.set_xlim(1.9, 8.7)
    ymin = min(-3.0, min(row["sign"] * float(row["spanning"]["fmb_t_conditional"]) for row in estimable) - 0.4)
    ax.set_ylim(ymin, 7.0)
    ax.set_xlabel("Sign-oriented univariate Fama--MacBeth $t$", fontsize=8.8)
    ax.set_ylabel("Sign-oriented conditional CZ $t$", fontsize=8.8)
    ax.set_title("Published-predictor spanning among all-model-alpha survivors", fontsize=9.8, loc="left")
    ax.text(
        0.985, 0.035,
        f"{len(passed)}/103 pass; {unavailable} non-estimable",
        ha="right", va="bottom", transform=ax.transAxes, fontsize=7.6, color="#34474D",
        bbox={"facecolor": "white", "edgecolor": "#CCD5D8", "boxstyle": "round,pad=0.3"},
    )
    fig.tight_layout()
    save_figure(fig, "fig5_spanning")
    return len(passed), unavailable


def subsample_figure(rows: list[dict]) -> dict[str, int]:
    population = [row for row in rows if row["passed_to_decay"]]
    assert len(population) == 325
    prepared = []
    for row in population:
        values = np.asarray(
            [row["sign"] * float(row["subsample"]["results"][key]) for key, _ in WINDOWS],
            dtype=float,
        )
        prepared.append((int(row["subsample"]["n_robust"]), float(values.mean()), values, row))
    prepared.sort(key=lambda item: (item[0], item[1]), reverse=True)
    matrix_array = np.vstack([item[2] for item in prepared])

    robust_counts = {n: sum(item[0] == n for item in prepared) for n in range(4, 8)}
    assert robust_counts == {4: 17, 5: 87, 6: 143, 7: 78}
    window_counts = {
        key: int(sum(item[2][j] > 1.96 for item in prepared))
        for j, (key, _) in enumerate(WINDOWS)
    }
    expected_window_counts = {
        "full": 325, "pre_anomaly": 275, "post_publication": 307,
        "pre_2000": 307, "post_2000": 253, "post_2010": 115,
        "ex_recession": 325,
    }
    assert window_counts == expected_window_counts

    decay_labels = [item[3]["decay"]["classification"] for item in prepared]
    decay_counts = {label: decay_labels.count(label) for label in ("stable", "strengthening", "decaying")}
    assert decay_counts == {"stable": 201, "strengthening": 6, "decaying": 118}
    decay_code = {"decaying": 0, "stable": 1, "strengthening": 2}
    decay_array = np.asarray([[decay_code[label]] for label in decay_labels], dtype=float)

    cmap = LinearSegmentedColormap.from_list(
        "signal_robustness", ["#A4473C", "#F3E6DF", "#FFFFFF", "#C9E1DC", "#146F68"]
    )
    decay_cmap = LinearSegmentedColormap.from_list(
        "decay_class", ["#B85A4F", "#AFC6BE", "#1B776B"], N=3
    )
    fig, (ax, decay_ax) = plt.subplots(
        1, 2, figsize=(7.25, 5.65), sharey=True,
        gridspec_kw={"width_ratios": [7.0, 0.42], "wspace": 0.045},
    )
    fig.patch.set_facecolor("white")
    image = ax.imshow(matrix_array, cmap=cmap, vmin=-3.0, vmax=7.0, aspect="auto", interpolation="nearest")
    decay_ax.imshow(decay_array, cmap=decay_cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")

    xlabels = [f"{label}\n{window_counts[key]}/325" for key, label in WINDOWS]
    ax.set_xticks(range(len(WINDOWS)), xlabels, fontsize=7.0)
    ax.tick_params(axis="x", length=0, pad=4)
    decay_ax.set_xticks([0], ["Decay\nclass"], fontsize=7.0)
    decay_ax.tick_params(axis="x", length=0, pad=4)

    group_ticks = []
    group_labels = []
    cursor = 0
    for n in (7, 6, 5, 4):
        count = robust_counts[n]
        group_ticks.append(cursor + (count - 1) / 2)
        group_labels.append(f"{n}/7  ({count})")
        cursor += count
        if cursor < len(prepared):
            ax.axhline(cursor - 0.5, color="white", linewidth=1.4)
            decay_ax.axhline(cursor - 0.5, color="white", linewidth=1.4)
    ax.set_yticks(group_ticks, group_labels, fontsize=7.1)
    ax.set_ylabel("Number of robust windows (signals)", fontsize=8.1)
    decay_ax.tick_params(axis="y", left=False, labelleft=False)
    for target in (ax, decay_ax):
        for spine in target.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(image, ax=[ax, decay_ax], fraction=0.025, pad=0.025)
    cbar.set_label("Sign-oriented Fama--MacBeth $t$", fontsize=7.7)
    cbar.ax.tick_params(labelsize=6.8)
    ax.set_title("Subsample robustness and temporal decay", fontsize=9.8, loc="left")
    fig.text(
        0.5, 0.018,
        "Rows are sorted by robust-window count and mean sign-oriented statistic.  "
        "Decay strip: red = decaying, gray = stable, green = strengthening.",
        ha="center", fontsize=6.9, color="#536166",
    )
    fig.subplots_adjust(left=0.13, right=0.89, top=0.92, bottom=0.13)
    save_figure(fig, "fig6_subsample")
    return decay_counts


def main() -> None:
    rows = collect_rows()
    counts = factor_alpha_figure(rows)
    spanning_pass, unavailable = spanning_figure(rows)
    decay_counts = subsample_figure(rows)
    print(
        "factor gate population=207 "
        + " ".join(f"{model}={count}" for model, count in counts.items())
        + f" all3=103"
    )
    print(f"CZ gate population=103 pass={spanning_pass} nonestimable={unavailable}")
    print(
        "temporal gate population=325 "
        + " ".join(f"{label}={count}" for label, count in decay_counts.items())
    )


if __name__ == "__main__":
    main()
