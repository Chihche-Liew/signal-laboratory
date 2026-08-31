#!/usr/bin/env python3
"""Generate the Section 6 post-search attrition and finalist tables.

The script treats the six formal runs as separate testing families. It reports
cumulative gate intersections within each run's horse-race survivor set and
validates the terminal rows against the frozen collaborator survivor bundle.

Outputs
-------
paper/tables.tex (the tagged Section 6 block only)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
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
EXPECTED_FINAL_COUNTS = {"B1": 2, "B2": 2, "B3": 2, "A1": 4, "A2": 1, "A3": 0}


def read_json(path: Path):
    return json.loads(path.read_text())


def tex(value: object) -> str:
    return (
        str(value)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def readable_signal(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def display_pair(value: str) -> str:
    labels = {
        "profit": "Profitability", "value": "Valuation", "invest": "Investment",
        "accrual": "Accruals", "quality": "Quality", "finance": "Financing",
        "intangible": "Intangibles", "distress": "Distress",
    }
    left, right = value.split("_", 1)
    return f"{labels[left]} x {labels[right]}"


def exact_named_row(rows: list[dict], name: str, t_stat: float) -> dict:
    named = [row for row in rows if row.get("name") == name]
    assert named, f"no row for {name}"
    if len(named) == 1:
        return named[0]
    ranked = sorted(named, key=lambda row: abs(float(row["t_stat"]) - float(t_stat)))
    assert len(ranked) == 1 or abs(float(ranked[0]["t_stat"]) - float(t_stat)) != abs(
        float(ranked[1]["t_stat"]) - float(t_stat)
    )
    return ranked[0]


def gate_rows() -> tuple[dict[str, dict[str, int]], set[str]]:
    """Return cumulative counts and the terminal signal-id set."""
    counts_by_run = {}
    terminal_ids = set()
    for code, dirname in RUNS:
        root = POSTHOC_DIR / dirname
        multiple = read_json(root / "multiple_testing" / "results.json")
        mt_rows = multiple["multiple_testing"]["per_signal"]
        bayes_rows = multiple["bayesian"]["signals"]
        bootstrap = read_json(root / "double_bootstrap" / "results.json")
        hurdle = float(bootstrap["double_bootstrap"]["t_hurdle_005"])
        alpha_rows = read_json(root / "multi_model_alpha" / "results.json")
        subsample_payload = read_json(root / "subsample" / "results.json")
        subsample = {row["signal_id"]: row for row in subsample_payload["subsample"]}
        decay = {row["signal_id"]: row for row in subsample_payload["decay"]}
        spanning = {row["signal_id"]: row for row in read_json(root / "spanning" / "results.json")}
        assert len(alpha_rows) == len(subsample) == len(decay) == len(spanning)

        counts = {
            "horse": 0, "bhy": 0, "bayes": 0, "bootstrap": 0,
            "subsample": 0, "nondecay": 0, "alpha": 0, "spanning": 0,
        }
        for alpha in alpha_rows:
            signal_id = alpha["signal_id"]
            full_t = float(alpha["full_fmb_t"])
            horse_t = float(alpha["horse_race_t"])
            mt_row = exact_named_row(mt_rows, alpha["name"], full_t)
            bayes_row = exact_named_row(bayes_rows, alpha["name"], full_t)
            span = spanning[signal_id]

            passed = math.isfinite(horse_t) and abs(horse_t) >= 1.96 and horse_t * full_t > 0
            counts["horse"] += int(passed)
            passed = passed and mt_row.get("bhy_reject") is True
            counts["bhy"] += int(passed)
            passed = passed and float(bayes_row["bayes_p"]["bayes_p_020"]) < 0.05
            counts["bayes"] += int(passed)
            passed = passed and abs(full_t) >= hurdle
            counts["bootstrap"] += int(passed)
            passed = passed and int(subsample[signal_id]["n_robust"]) >= 4
            counts["subsample"] += int(passed)
            passed = passed and decay[signal_id]["classification"] != "decaying"
            counts["nondecay"] += int(passed)
            alpha_pass = all(
                alpha.get(f"{model}_status") == "ok"
                and alpha.get(f"{model}_direction_consistent") is True
                and alpha.get(f"{model}_survives_196") is True
                for model in ("FF5", "FF6", "q4")
            )
            passed = passed and alpha_pass
            counts["alpha"] += int(passed)
            span_pass = (
                span.get("status") == "ok"
                and span.get("direction_consistent") is True
                and span.get("survives_196") is True
            )
            passed = passed and span_pass
            counts["spanning"] += int(passed)
            if passed:
                terminal_ids.add(signal_id)

        assert counts["horse"] == len(alpha_rows)
        assert all(left >= right for left, right in zip(counts.values(), list(counts.values())[1:]))
        assert counts["spanning"] == EXPECTED_FINAL_COUNTS[code]
        counts_by_run[code] = counts
    return counts_by_run, terminal_ids


def attrition_table(counts_by_run: dict[str, dict[str, int]]) -> str:
    rows = []
    for code, _ in RUNS:
        c = counts_by_run[code]
        rows.append(
            f"{code} & {c['horse']} & {c['bhy']} & {c['bayes']} & {c['bootstrap']} & "
            f"{c['subsample']} & {c['nondecay']} & {c['alpha']} & {c['spanning']} \\\\"
        )
    totals = {key: sum(counts_by_run[code][key] for code, _ in RUNS) for key in counts_by_run["B1"]}
    rows.append(
        f"Total & {totals['horse']} & {totals['bhy']} & {totals['bayes']} & {totals['bootstrap']} & "
        f"{totals['subsample']} & {totals['nondecay']} & {totals['alpha']} & {totals['spanning']} \\\\"
    )
    return r"""
% ----------------------------------------------------------------------
% Table: cumulative post-search attrition
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Cumulative Post-Search Attrition Within the Horse-Race Set}
\label{tab:posthoc_attrition}
\scriptsize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3.7pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Run & Horse & BHY & Bayes & Double boot. & Subsamples & Non-decay & Alpha 3/3 & CZ \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\midrule
""" + rows[-1] + r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\scriptsize
Counts are cumulative intersections within each run's final within-pair
horse-race set. BHY is the run-level false-discovery-rate screen; Bayes requires
posterior null probability below 5\% under the moderate prior; double bootstrap
requires the run-specific 5\% hurdle; subsamples requires significance in at
least four of seven windows; non-decay excludes signals classified as decaying;
Alpha 3/3 requires direction-consistent significance under FF5, FF6, and the
$q$-factor model; CZ requires a direction-consistent conditional statistic above
1.96 against the selected Chen--Zimmermann controls. The six runs remain
separate testing families; the total row is descriptive and does not pool their
multiple-testing calculations. WLS is a reported diagnostic, not a gate.
\end{flushleft}
\end{table}
"""


def finalist_table(catalog: list[dict]) -> str:
    rows = []
    for record in catalog:
        alpha = record["evidence"]["horse_and_alpha"]
        sub = record["evidence"]["subsample"]
        span = record["evidence"]["spanning"]
        min_alpha_t = min(abs(float(alpha[f"{model}_talpha"])) for model in ("FF5", "FF6", "q4"))
        rows.append(
            f"{record['code']} & {tex(readable_signal(record['name']))} & "
            f"{tex(display_pair(record['pair']))} & {record['generation']} & "
            f"{float(alpha['full_fmb_t']):.2f} & {float(alpha['horse_race_t']):.2f} & "
            f"{sub['n_robust']}/{sub['n_available']} & {min_alpha_t:.2f} & "
            f"{float(span['fmb_t_conditional']):.2f} \\\\"
        )
    return r"""
% ----------------------------------------------------------------------
% Table: full-pipeline signals
% ----------------------------------------------------------------------

\begin{table}[htbp]
\centering
\caption{Signals Surviving the Complete Post-Search Protocol}
\label{tab:full_pipeline_signals}
\scriptsize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{2.25cm}rrrrrr@{}}
\toprule
Run & Signal & Theme pair & Gen. & FMB $t$ & Horse $t$ & Robust & Min. $|t^\alpha|$ & CZ $t$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{flushleft}
\scriptsize
These 11 signals pass every cumulative gate in
Table~\ref{tab:posthoc_attrition}. Robust is the number of significant windows
out of seven. Min. $|t^\alpha|$ is the smallest absolute alpha statistic across
FF5, FF6, and the $q$-factor model, all of which have the declared direction.
CZ $t$ is the conditional Fama--MacBeth statistic after controlling for the
selected published predictors. Appendix Table~\ref{tab:full_pipeline_expressions}
reports the executable expressions. Signal numbers used in Section~\ref{sec:reasoning}
follow this row order.
\end{flushleft}
\end{table}
"""


def expression_table(catalog: list[dict]) -> str:
    rows = []
    for index, record in enumerate(catalog, start=1):
        expression = tex(record["expression"]).replace(", ", ",\\allowbreak{} ").replace("(", "(\\allowbreak{}")
        rows.append(
            f"S{index} & {record['code']} & {tex(readable_signal(record['name']))} & "
            f"{record['expected_sign']} & \\texttt{{{expression}}} \\\\"
        )
    return r"""
% ----------------------------------------------------------------------
% Table: full-pipeline expressions (appendix)
% ----------------------------------------------------------------------

\clearpage
\begingroup
\scriptsize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}ll>{\raggedright\arraybackslash}p{3.2cm}l>{\raggedright\arraybackslash}p{8.1cm}@{}}
\caption{Expressions of the Full-Pipeline Signals}
\label{tab:full_pipeline_expressions} \\
\toprule
ID & Run & Signal & Sign & Expression \\
\midrule
\endfirsthead
\multicolumn{5}{l}{\textit{Table~\ref{tab:full_pipeline_expressions} continued}} \\[4pt]
\toprule
ID & Run & Signal & Sign & Expression \\
\midrule
\endhead
\midrule
\multicolumn{5}{r}{\textit{Continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(rows) + r"""
\end{longtable}
\begin{flushleft}
\scriptsize
Expressions are reproduced exactly from the final evaluated proposals. Signal
IDs S1--S11 match Table~\ref{tab:full_pipeline_signals} and the trajectory case
studies in Section~\ref{sec:reasoning}.
\end{flushleft}
\endgroup
"""


def write_tables(block: str) -> None:
    path = PAPER_DIR / "tables.tex"
    begin = "% BEGIN SECTION6 TABLES"
    end = "% END SECTION6 TABLES"
    tagged = begin + "\n" + block + "\n" + end
    current = path.read_text() if path.exists() else ""
    if begin in current and end in current:
        prefix, rest = current.split(begin, 1)
        _, suffix = rest.split(end, 1)
        current = prefix.rstrip() + "\n\n" + tagged + suffix
    else:
        current = current.rstrip() + "\n\n" + tagged + "\n"
    path.write_text(current)


def main() -> None:
    catalog = read_json(CATALOG_PATH)
    assert len(catalog) == 11
    counts_by_run, terminal_ids = gate_rows()
    catalog_ids = {record["signal_id"] for record in catalog}
    assert terminal_ids == catalog_ids
    write_tables(
        attrition_table(counts_by_run)
        + finalist_table(catalog)
        + expression_table(catalog)
    )
    for code, _ in RUNS:
        c = counts_by_run[code]
        print(
            f"{code}: horse={c['horse']} bhy={c['bhy']} bayes={c['bayes']} "
            f"bootstrap={c['bootstrap']} subsample={c['subsample']} "
            f"nondecay={c['nondecay']} alpha={c['alpha']} cz={c['spanning']}"
        )
    print(f"validated {len(catalog)} complete-protocol signals against the frozen bundle")


if __name__ == "__main__":
    main()
