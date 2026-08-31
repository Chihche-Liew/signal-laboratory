# SignalLab

SignalLab is a research system in which specialized LLM agents run the hypothesis-discovery
loop in empirical asset pricing inside an environment that humans froze in advance. This
repository is the public code and replication release accompanying the paper.

The agents choose which economic ideas to pursue. They cannot change how those ideas are
tested: the hypothesis language, the sample, the evaluation pipeline, and the significance
protocol are fixed before the search begins. Because every hypothesis must be expressed in
that language and pass through that pipeline, the entire search population is recorded —
including the failures — and enters the statistical inference rather than sitting outside it.

## What the study found

Across six runs the system evaluated 9,656 hypotheses. The headline is the attrition, not the
yield:

| Stage | Records |
|---|---|
| Evaluated hypotheses (per primary run) | 1,603–1,607 |
| Clear a signed significance screen | 59.1%–62.3% |
| Survive within-task horse races (all six runs) | 449 |
| Clear multiplicity, stability, and factor gates | 103 |
| Retain conditional content against published predictors | **11** |

High first-pass yield is a property of an adaptive in-sample search, not evidence. The
inferential weight rests on the cumulative post-search protocol, and four of the eleven
survivors rest on 24–28 month common samples — a limitation the artifact records rather than
buries.

## Two ways to read this repository

**If you want the tool**, go to [Installation](#installation) and `src/siglab/`.

**If you want the research record**, start at [`paper/ara/PAPER.md`](paper/ara/PAPER.md). It is
an [Agent-Native Research Artifact](https://arxiv.org/abs/2604.24658): the paper's claims,
evidence bindings, and exploration graph in machine-readable form, written so an agent can
check each claim against the shipped artifacts without reading the PDF.

```
paper/ara/
  PAPER.md                    # manifest, layer index, reproduction boundary
  logic/problem.md            # observations -> gaps -> design insight
  logic/claims.md             # 9 claims bound to shipped evidence, 5 untestable assumptions
  logic/experiments.md        # 9 experiment specs that verify them
  src/environment.md          # runtime, credentials, why reruns do not match
  src/artifacts.md            # pointer index to the codebase
  trace/exploration_tree.yaml # 4 survivor lineages, dead ends quoted verbatim
  evidence/README.md          # every claim mapped to the run artifacts that ground it
```

The artifact copies no data. Its `src/` and `evidence/` layers are index files pointing at the
code and run artifacts already in this repository.

Every count in `paper/ara/logic/claims.md` recomputes from `paper/data/` and `paper/generated/`
with no network access, no model calls, and no licensed data. What cannot be reproduced without
a CRSP/Compustat/WRDS subscription is an end-to-end discovery rerun. Note also that LLM search
is stochastic and path dependent: even with full access, a rerun produces a different search
path and different final hypotheses. That is a reported finding, not a reproduction failure.

## What is in the release

- the reusable Python package in `src/siglab/`;
- experiment runners in `scripts/`;
- the four frozen run configurations in `paper/configs/`, which produced the six formal runs;
- the completed run artifacts for those six runs in `paper/data/` (~105 MB);
- adjudicated mechanism and critique labels in `paper/generated/`;
- the scripts and released figures in `paper/scripts/` and `paper/figures/`;
- the agent-native research artifact in `paper/ara/`.

Run codes used throughout: `B1`/`B2`/`B3` are three independently initialized primary runs;
`A1` removes critique and revision; `A2` and `A3` substitute alternative proposers. Each run is
a separate testing family, and each comparison configuration was executed once — their counts
characterize realized output under a configuration and must not be used to rank configurations.

Paper source files are intentionally not part of this repository. Licensed CRSP, Compustat, and
WRDS source data are also excluded.

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[paper]'
```

Copy `.env.example` to `.env` and fill in only the credentials you need. Never commit `.env`.
Verifying claims and regenerating figures and tables need **no credentials at all**. A discovery
rerun needs `WRDS_USERNAME` and a model-provider key; the two `ablation_gpt_*` configurations
additionally need an OpenAI credential and `pip install -e '.[openai]'`, which `.env.example`
does not yet template. See [`paper/ara/src/environment.md`](paper/ara/src/environment.md).

## Replication

See [`paper/README.md`](paper/README.md) for the exact figure and table regeneration commands.
Run the scripts in the order given there — the first creates the `paper/tables.tex` that the
later ones append to. The commands operate on the frozen artifacts in this release and make no
model or data-provider calls.

For a targeted discovery run after configuring data and a model provider:

```bash
python scripts/experiments/run_lab.py \
  --config paper/configs/baseline.yaml \
  --pair accrual_quality
```

## Repository scope

A clean public release with a new Git history. It excludes tests, internal project
documentation, private workflow materials, paper source, local caches, and restricted raw data.
Paths inside the shipped run artifacts were rewritten from the original private run layout to
this release's layout so that every reference resolves.

## Citation

Joyce Liu, Miao Liu, Zhizhe Liu, and Danqing Mei. *Can AI Do Financial Research? LLM-Guided
Hypothesis Discovery in Asset Pricing.* Working paper, 2026.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
