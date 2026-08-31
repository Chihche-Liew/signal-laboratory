# Environment

- **Language/runtime**: Python >= 3.10 (`requires-python = ">=3.10"` in `pyproject.toml`).
- **Install**: `pip install -e '.[paper]'` from the repository root.
- **Framework**: pandas / numpy / statsmodels. No deep-learning framework.
- **Hardware**: commodity CPU, no GPU. The replication scripts in `paper/scripts/` run on a
  laptop; an end-to-end discovery rerun is bounded by model-provider latency, not local compute.
- **Key dependencies**: declared as lower bounds in `pyproject.toml`, which is authoritative —
  `numpy>=1.24`, `pandas>=2.0`, `scipy>=1.10`, `statsmodels>=0.14`, `wrds>=3.1`, `pyyaml>=6.0`,
  `pyarrow>=12.0`, `pandas-datareader>=0.10`, `python-dotenv>=1.0`, `anthropic>=0.25`,
  `threadpoolctl>=3.1`, `tqdm>=4.60`. Extra `paper` adds `matplotlib>=3.7`. Extras `openai`
  (`openai>=1.40`) and `gemini` (`google-genai>=1.0`) are optional and are **not** installed by
  `'.[paper]'`. No exact-version lockfile is published; these floors are what the release
  supports.

## Data sources

- **CRSP, Compustat, and linked WRDS tables** — licensed, accessed through the `wrds` client
  with a `WRDS_USERNAME` credential. Not redistributable and not included in this repository.
  Sample as used: 1963-2024, financial firms and microcaps excluded, NYSE breakpoints, value
  weights.
- **Frozen run artifacts** — `paper/data/`. The portable outputs of the licensed pipeline, and
  what every claim in `logic/claims.md` is checked against. They require no credentials.

## Credentials

Copy `.env.example` to `.env` (gitignored). What you need depends on what you are doing:

| Task | Credentials required |
|---|---|
| Verify claims against shipped evidence | none |
| Regenerate figures and tables (`paper/scripts/*.py`) | none |
| Rerun discovery on `baseline.yaml` or `ablation_no_critique.yaml` | `WRDS_USERNAME` + `ANTHROPIC_API_KEY` |
| Rerun `ablation_gpt_5_5.yaml` or `ablation_gpt_5_6.yaml` | the above, plus an OpenAI credential and `pip install -e '.[openai]'` |

The last row is a gap in the shipped template: those two configs set
`proposer.llms.proposer.provider: openai` (models `gpt-5.5` and `gpt-5.6-sol`) with an
`anthropic` critic, but `.env.example` ships only `WRDS_USERNAME` and `ANTHROPIC_API_KEY`.

## Protocols

Frozen run specifications live in `paper/configs/`: `baseline.yaml` (three independent primary
executions, B1-B3), `ablation_no_critique.yaml` (A1), `ablation_gpt_5_5.yaml` (A2),
`ablation_gpt_5_6.yaml` (A3, producing the run directory `ablation_gpt_5_6_sol`). Each
comparison configuration was executed once.

## Random seeds, and why reruns will not match

No seed makes an end-to-end rerun reproducible. LLM search is stochastic and path dependent:
even with full data and credentials, rerunning `baseline.yaml` produces a different search path
and different final hypotheses. This is not a defect and not a reproduction failure —
expression-level non-recurrence is a reported finding (C07). What reproduces is the shape of
the discovery funnel and the economic mechanisms, not the formulas.

Everything downstream of the frozen artifacts is deterministic: the analysis scripts in
`paper/scripts/` recompute every reported number from `paper/data/` with no network access, no
model calls, and no licensed data.
