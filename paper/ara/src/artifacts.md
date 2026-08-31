# Codebase pointer index

The implementation persists in this repository and is **not copied into this artifact**. One
entry per component; all paths are relative to the repository root. Run records are not indexed
here — they are evidence, and live in [`../evidence/README.md`](../evidence/README.md).

## Discovery loop — `src/siglab/lab/`
- **Files**: `loop.py`, `runner.py`, `runspec.py`, `archive.py`, `recorder.py`, `task.py`,
  `budget.py`, `validation.py`, `manifest.py`, `posthoc.py`, `lesson.py`, and the subpackages
  `proposer/` (`proposer_critic.py`, `single_agent.py`, `debate.py`, `socratic.py`),
  `reflector/`, `stopping/`, `prompting/`, `parsing/`, `context/`, `evaluator/`, `llm/`
- **Nature**: the generation loop — proposal, critique, revision, archive, stopping
- **What it does**: runs one theme-pair task for N generations; writes `archive.json` and
  `generations/gen_NNN/{critic_notes.json,raw_responses.jsonl}`
- **How to run**: `python scripts/experiments/run_lab.py --config <cfg> --pair <pair>`
- **Claims supported**: C01, C02, C04, C08

## Symbolic hypothesis language — `src/siglab/agent/`
- **Files**: `variables.py`, `operators.py`, `themes.py`, `executor.py`, `signal_agent.py`,
  `signal_evaluator.py`
- **Nature**: the frozen search space — Compustat variables, operator set, expression executor
- **What it does**: bounds what an agent may propose, and evaluates a proposed expression
- **Claims supported**: C01

## Post-search gates — `src/siglab/assay/`
- **Files**: `multiple_testing.py`, `double_bootstrap.py`, `subsample.py`, `spanning.py`,
  `horse_race.py`, `sample.py`
- **Nature**: the cumulative validation protocol
- **How to run**: `scripts/experiments/run_posthoc.py`, and the per-gate entry points
  `run_multiple_testing.py`, `run_double_bootstrap.py`, `run_subsample.py`,
  `run_spanning_test.py`, `run_multi_model_alpha.py`, `run_horse_race_selection.py`
- **Claims supported**: C05, C06, C09

## Asset-pricing machinery — `src/siglab/factor_model/`, `src/siglab/portfolio/`
- **Files**: `fama_macbeth.py`, `grs.py`, `models.py`, `time_series.py`; `sorts.py`,
  `breakpoints.py`, `returns.py`, `mve.py`
- **Nature**: Fama-MacBeth, GRS, portfolio sorts, NYSE breakpoints, value weights
- **Claims supported**: C01, C03, C06

## Data access — `src/siglab/data/`
- **Files**: `crsp.py`, `compustat.py`, `wrds_conn.py`, `merge.py`, `factors.py`,
  `cz_anomalies.py`, `cache.py`
- **Nature**: licensed-data ingestion; requires a WRDS subscription, see
  [`environment.md`](environment.md)
- **Claims supported**: C06 — the published-predictor controls come from `cz_anomalies.py`

## Frozen configurations — `paper/configs/`
- **Files**: `baseline.yaml` (B1/B2/B3), `ablation_no_critique.yaml` (A1),
  `ablation_gpt_5_5.yaml` (A2), `ablation_gpt_5_6.yaml` (A3)
- **Note**: `ablation_gpt_5_6.yaml` produces the run directory `ablation_gpt_5_6_sol`
- **Claims supported**: C01, C03, C04

## Analysis scripts — `paper/scripts/`
- **Files**: `section3_generate_framework_figure.py`, `section4_generate_tables.py`,
  `section5_generate_results.py`, `section6_generate_figures.py`,
  `section6_generate_results.py`, `section7_generate_results.py`
- **Nature**: recompute every reported number from `paper/data/`; no network, no model calls
- **Claims supported**: C01-C09
