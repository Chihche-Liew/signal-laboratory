# Evidence Index

Every claim in `logic/claims.md` is grounded in run artifacts that ship in this repository but
deliberately live **outside** this artifact directory: roughly 105 MB across six runs x 28
theme-pair tasks (168 pair directories) under `paper/data/formal/`, plus adjudicated label
files under `paper/generated/` and released figures under `paper/figures/`. They are shipped
once, for the replication scripts in `paper/scripts/`, and are not duplicated here. This index
maps each body of evidence to the claims it grounds and the experiment that consumes it.

All paths are relative to the repository root.

Run codes: `B1`=`baseline_r1`, `B2`=`baseline_r2`, `B3`=`baseline_r3`,
`A1`=`ablation_no_critique`, `A2`=`ablation_gpt_5_5`, `A3`=`ablation_gpt_5_6_sol`.
Each run is a separate testing family; totals across runs are descriptive only.

## Primary run artifacts

| Path | What it holds | Grounds | Used by |
|---|---|---|---|
| `paper/data/formal/<run>/<pair>/archive.json` | every evaluated hypothesis; `evaluated[].fmb_tstat`, `.expected_sign`, `.generation` | C01, C02, C03, C07 | E01, E02, E03, E07 |
| `paper/data/formal/<run>/<pair>/selection/horse_race.json` | within-task horse-race selection; `n_survivors` | C03, C07 | E03, E07 |
| `paper/data/formal/<run>/<pair>/generations/gen_NNN/critic_notes.json` | every critic judgment: `verdict`, `score`, `rationale` | C04, C08 | E04, E08 |
| `paper/data/formal/<run>/<pair>/generations/gen_NNN/raw_responses.jsonl` | proposer / critic / revision rounds verbatim | C08 | E08 |
| `paper/data/formal/posthoc/<run>/multiple_testing/` | Benjamini-Yekutieli gate | C05 | E05 |
| `paper/data/formal/posthoc/<run>/double_bootstrap/` | signed double-bootstrap gate | C05 | E05 |
| `paper/data/formal/posthoc/<run>/subsample/` | seven-window stability gate | C05 | E05 |
| `paper/data/formal/posthoc/<run>/multi_model_alpha/` | FF5 / FF6 / HXZ q alpha gate | C05 | E05 |
| `paper/data/formal/posthoc/<run>/spanning/` | conditional tests against published predictors; `fmb_t_univariate`, `fmb_t_conditional`, `n_valid_months`, `n_controls`, `skipped_controls`, `direction_consistent` | C05, C06, C09 | E05, E06, E09 |
| `paper/data/signal_catalog.json` | the 11 complete-protocol survivors with per-gate evidence, and `trajectory_rel` into the bundles below | C05, C09 | E05, E09 |
| `paper/data/trajectories/<code>/<signal_id>_<name>/` | per-survivor bundle: the producing generation's prompts, responses, full proposal batch, critic notes, run manifest, selection and post-hoc extracts | C08, C09 | E08 |

## Adjudicated labels

| Path | What it holds | Grounds | Used by |
|---|---|---|---|
| `paper/generated/mechanism_codes.csv` | economic-mechanism labels per hypothesis | C04, C07 | E04, E07 |
| `paper/generated/mechanism_coding_agreement.json` | inter-coder reliability for the above | C04 | E04 |
| `paper/generated/critique_codes.csv` | critique-taxonomy labels | C04 | E04 |
| `paper/generated/critique_codebook.json` | the critique codebook | C04 | E04 |
| `paper/generated/critique_coding_metrics.json` | reliability for the critique coding | C04 | E04 |
| `paper/generated/section7_critique_summary.json` | aggregated critique statistics | C04 | E04 |

These label files are **model-produced coding of model output** (`ai-suggested`): outcome-blind,
but not human-adjudicated end to end. Read them only alongside their reliability files. The
assumption that this coding is a valid taxonomy is recorded as U02.

## Released figures

| Path | Grounds |
|---|---|
| `paper/figures/fig1_generation_dynamics.{png,pdf}` | C02 |
| `paper/figures/fig2_pair_frontier.{png,pdf}` | C03 |
| `paper/figures/fig3_architecture_contrasts.{png,pdf}` | C03 |
| `paper/figures/fig4_multi_model.{png,pdf}` | C05 |
| `paper/figures/fig5_spanning.{png,pdf}` | C06 |
| `paper/figures/fig6_subsample.{png,pdf}` | C05 |
| `paper/figures/fig7_critique_taxonomy.{png,pdf}` | C04 |
| `paper/figures/fig8_critique_novelty.{png,pdf}` | C04 |

`fig0_workflow` illustrates the framework and grounds no claim.

## What is not here

- **Raw CRSP / Compustat / WRDS panels.** Licensed, not redistributable. Every number above is
  computed from them, but they cannot ship. See `../src/environment.md`.
- **`paper/tables.tex`.** A generated replication output, gitignored. It is created by the
  `section4` step of the documented sequence in `paper/README.md`; see E05 `Caveat (execution)`.
- **A blind-reproduction split.** The ARA schema separates `evidence/` so a verifying agent can
  receive `logic/` and `src/` while ground truth is withheld. This artifact ships the full
  evidence layer by design — it is an audit record, not a blind-reproduction target. See the
  reproduction boundary in `../PAPER.md`.
