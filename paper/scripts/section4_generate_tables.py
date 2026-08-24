#!/usr/bin/env python3
"""Generate the Section 3/4 table block from code and configuration.

All generated tables are written to ``paper/tables.tex``. Section and appendix
prose refer to the table labels but do not input individual table files:

  tab:operators             operator classes of the hypothesis language
                            (from src/siglab/agent/operators.py)
  tab:agent_configurations  the frozen experiment matrix
                            (from paper/configs/*.yaml)
  tab:theme_pairs           the 28 cross-theme tasks, intended for the appendix
                            (from src/siglab/lab/tasks/cross_theme.py)

Run from any working directory:

    python paper/scripts/section4_generate_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
CONFIG_DIR = PAPER_DIR / "configs"
sys.path.insert(0, str(REPO_ROOT / "src"))

from siglab.agent.operators import OPERATOR_CATALOG  # noqa: E402
from siglab.lab.tasks.cross_theme import CROSS_THEME_TASKS  # noqa: E402


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


# ----------------------------------------------------------------------
# Table 1: operator classes
# ----------------------------------------------------------------------

OPERATOR_CLASSES = [
    (
        "Scaling and change",
        ["RATIO", "GROWTH", "DELTA"],
        "Ratios of accounting items; year-over-year growth and level changes.",
    ),
    (
        "Time-series state",
        ["MA", "TREND", "VOL", "ACCEL", "LAG", "TS_MIN", "TS_MAX", "TS_SUM", "TS_COUNT", "TS_RANK"],
        "Within-firm smoothing, slope, volatility, acceleration, lags, "
        "distance from own extremes, cumulative flows, reporting persistence.",
    ),
    (
        "Composition",
        ["ADD", "SUB", "MUL", "SUMRANK", "INDICATOR", "COALESCE"],
        "Arithmetic interaction, rank composites, threshold indicators, "
        "economically justified missing-value fallback.",
    ),
    (
        "Mathematical transforms",
        ["ABS", "SIGN", "LOG", "INV", "NEG", "MIN", "MAX"],
        "Nonlinear reshaping and weakest-link or strongest-link constructions.",
    ),
    (
        "Cross-sectional and industry-relative",
        ["RANK", "ZSCORE", "WINSOR", "IND_ADJ", "IND_ZSCORE", "IND_RANK"],
        "Cross-sectional ranking, standardization, winsorization; "
        "industry-median adjustment and industry-relative scores.",
    ),
]


def signature(name: str) -> str:
    op = OPERATOR_CATALOG[name]
    args = ["a", "b"][: op.arity] if op.arity == 2 else ["x"]
    args += list(op.params)
    return rf"\texttt{{{tex(name)}({', '.join(args)})}}"


def operator_table() -> str:
    covered = [n for _, names, _ in OPERATOR_CLASSES for n in names]
    assert sorted(covered) == sorted(OPERATOR_CATALOG), (
        set(covered) ^ set(OPERATOR_CATALOG)
    )
    rows = []
    for label, names, purpose in OPERATOR_CLASSES:
        sigs = ", ".join(signature(n) for n in names)
        rows.append(rf"{label} ({len(names)}) & {sigs} & {purpose} \\")
    body = "\n".join(rows)
    return rf"""
% ----------------------------------------------------------------------
% Table: operator classes of the hypothesis language
% ----------------------------------------------------------------------

\begin{{table}}[htbp]
\centering
\caption{{Operator Classes of the Hypothesis Language}}
\label{{tab:operators}}
\footnotesize
\renewcommand{{\arraystretch}}{{1.15}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{3.1cm}} >{{\raggedright\arraybackslash}}X >{{\raggedright\arraybackslash}}p{{4.6cm}}@{{}}}}
\toprule
Class (count) & Operators & Economic role \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\begin{{flushleft}}
\footnotesize
The language contains {len(OPERATOR_CATALOG)} operators with explicit arity and
parameter domains. Arguments $a$, $b$, and $x$ denote nested sub-expressions or
variables; named arguments are integer parameters (lag, window, threshold, or
tail percentile). Operators compose recursively, so the admissible space contains
ratios of flows and stocks, temporal states, industry-relative measures, and rank
composites of any of these.
\end{{flushleft}}
\end{{table}}
"""


# ----------------------------------------------------------------------
# Table 2: agent configurations (frozen experiment matrix)
# ----------------------------------------------------------------------

ROWS = [
    ("Primary", "baseline.yaml", 3),
    ("Single-agent", "ablation_no_critique.yaml", 1),
    ("GPT-5.5 proposer", "ablation_gpt_5_5.yaml", 1),
    ("GPT-5.6-sol proposer", "ablation_gpt_5_6.yaml", 1),
]

PROVIDER_LABEL = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google"}


def load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def llm_label(spec: dict, role: str) -> str:
    proposer = spec["proposer"]
    role_spec = (proposer.get("llms") or {}).get(role)
    if role_spec is None:
        if role == "critic":
            return "---"
        role_spec = spec.get("llm", {})
    provider = role_spec.get("provider", spec.get("llm", {}).get("provider", "anthropic"))
    model = role_spec.get("model", spec.get("llm", {}).get("model", ""))
    breakable_model = tex(model).replace("-", r"-\allowbreak{}")
    return rf"\texttt{{{breakable_model}}}\newline ({PROVIDER_LABEL.get(provider, provider.title())})"


TOPOLOGY = {
    "proposer_critic": ("Propose--critique--revise", "2--3"),
    "single_agent": ("Propose", "1"),
}

FILTER_LABEL = {
    "exclude_financials": "financial firms",
    "exclude_microcap": "microcaps",
}


def agent_table() -> str:
    specs = [(label, load(filename), reps) for label, filename, reps in ROWS]

    # Verify the common design quantities directly against every YAML.
    base = specs[0][1]
    common_pairs = len(base["task"]["pairs"])
    common_gens = base["loop"]["n_generations"]
    common_n = base["proposer"]["n_proposals"]
    common_filters = tuple(base["evaluator"]["sample_filters"])
    common_threshold = base["evaluator"]["success_threshold"]
    common_alpha_model = base["evaluator"]["alpha_factor_model"]
    common_effort = base["llm"]["reasoning_effort"]
    for _, spec, _ in specs[1:]:
        assert len(spec["task"]["pairs"]) == common_pairs
        assert spec["loop"]["n_generations"] == common_gens
        assert spec["proposer"]["n_proposals"] == common_n
        assert tuple(spec["evaluator"]["sample_filters"]) == common_filters
        assert spec["evaluator"]["success_threshold"] == common_threshold
        assert spec["evaluator"]["alpha_factor_model"] == common_alpha_model
        assert spec["llm"]["reasoning_effort"] == common_effort

    rows = []
    for label, spec, reps in specs:
        topo, calls = TOPOLOGY[spec["proposer"]["type"]]
        rows.append(
            " & ".join(
                [label, llm_label(spec, "proposer"), llm_label(spec, "critic"), topo, calls, str(reps)]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    excluded = " and ".join(FILTER_LABEL[f] for f in common_filters)
    return rf"""
% ----------------------------------------------------------------------
% Table: agent configurations (frozen experiment matrix)
% ----------------------------------------------------------------------

\begin{{table}}[htbp]
\centering
\caption{{Agent Configurations}}
\label{{tab:agent_configurations}}
\footnotesize
\renewcommand{{\arraystretch}}{{1.15}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{2.3cm}} >{{\raggedright\arraybackslash}}p{{2.65cm}} >{{\raggedright\arraybackslash}}p{{2.65cm}} >{{\raggedright\arraybackslash}}X >{{\raggedright\arraybackslash}}p{{2.0cm}} c@{{}}}}
\toprule
Configuration & Proposer & Critic & Interaction & Calls/gen. & Runs \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\begin{{flushleft}}
\footnotesize
Every run covers the same {common_pairs} cross-theme tasks with {common_gens} generations per
task and {common_n} requested proposals per generation, uses {common_effort} reasoning effort
for every model role, excludes {excluded} from the evaluation
universe, applies the expected-sign Fama--MacBeth screen at $|t|>{common_threshold:.2f}$, and
reports the discovery-stage long--short alpha against the Fama--French
{'five' if common_alpha_model == 5 else common_alpha_model}-factor model. The configurations
differ only in the proposer model or in whether the critique--revision
subloop is present; the single-agent configuration therefore also makes fewer model
calls per generation. The primary configuration is executed in three independent runs;
each comparison configuration is executed once. Each run is its own search universe
and its own post-discovery testing family.
\end{{flushleft}}
\end{{table}}
"""


# ----------------------------------------------------------------------
# Table 3: the 28 cross-theme tasks (appendix)
# ----------------------------------------------------------------------

def first_sentence(text: str) -> str:
    head = text.split(". ")[0].rstrip(".")
    return head + "."


def theme_pair_table() -> str:
    assert len(CROSS_THEME_TASKS) == 28
    rows = []
    for spec in CROSS_THEME_TASKS:
        label = spec.interaction_label.replace(" x ", r" $\times$ ")
        rows.append(rf"{tex(label)} & {tex(first_sentence(spec.economic_story))} \\")
    body = "\n".join(rows)
    return rf"""
% ----------------------------------------------------------------------
% Table: the 28 cross-theme tasks (intended for the appendix)
% ----------------------------------------------------------------------

\clearpage
\begingroup
\scriptsize
\renewcommand{{\arraystretch}}{{1.12}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{longtable}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.23\textwidth}}p{{0.72\textwidth}}@{{}}}}
\caption{{Cross-Theme Tasks}}
\label{{tab:theme_pairs}} \\
\toprule
Theme pair & Economic objective supplied to the task \\
\midrule
\endfirsthead
\multicolumn{{2}}{{l}}{{\textit{{Table~\ref{{tab:theme_pairs}} continued}}}} \\[4pt]
\toprule
Theme pair & Economic objective supplied to the task \\
\midrule
\endhead
\midrule
\multicolumn{{2}}{{r}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{body}
\end{{longtable}}
\begin{{flushleft}}
\scriptsize
All $\binom{{8}}{{2}}=28$ pairwise combinations of the eight themes. The objective
shown is the first sentence of each task's economic story; the task also carries
example interaction ideas, theme-specific variable menus, and preferred operators,
which are conceptual anchors rather than templates. Each proposal must use at least
one variable exclusive to each theme of the pair.
\end{{flushleft}}
\endgroup
"""


def main() -> None:
    path = PAPER_DIR / "tables.tex"
    header = """% ======================================================================
%  paper/tables.tex
%  Auto-generated by the section-prefixed scripts in paper/scripts/.
%  Do not edit generated table blocks by hand.
% ======================================================================
"""
    begin = "% BEGIN SECTION4 TABLES"
    end = "% END SECTION4 TABLES"
    block = begin + "\n" + "".join(
        [operator_table(), agent_table(), theme_pair_table()]
    ) + "\n" + end

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


if __name__ == "__main__":
    main()
