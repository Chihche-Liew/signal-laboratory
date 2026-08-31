# SignalLab

SignalLab is a research system for generating and testing cross-theme
fundamental signals. This repository is the public code and replication
release accompanying the paper.

The release contains:

- the reusable Python package in `src/siglab/`;
- experiment runners in `scripts/`;
- the six frozen formal-run configurations in `paper/configs/`;
- the completed formal-run artifacts needed for the reported analyses in
  `paper/data/`;
- the scripts and derived inputs used to regenerate the paper's figures and
  tables in `paper/`;
- the agent-native research artifact in `paper/ara/`.

Paper source files are intentionally not part of this repository. Licensed
CRSP, Compustat, and WRDS source data are also excluded. The included completed
run artifacts are sufficient to regenerate the paper's reported figures and
tables; rerunning discovery or empirical tests from raw data requires the
corresponding data subscriptions and model credentials.

## Two ways to read this repository

**If you want the tool**, go to [Installation](#installation) and `src/siglab/`.

**If you want the research record**, start at [`paper/ara/PAPER.md`](paper/ara/PAPER.md). It is an
[Agent-Native Research Artifact](https://arxiv.org/abs/2604.24658): the paper's claims,
evidence bindings, and exploration graph in machine-readable form, written so that an agent
can check each claim against the shipped artifacts without reading the PDF.

```
paper/
  ara/
    PAPER.md                    # manifest, layer index, reproduction boundary
    logic/problem.md            # observations -> gaps -> design insight
    logic/claims.md             # 9 falsifiable claims, each bound to shipped evidence
    logic/experiments.md        # 9 experiment specs that verify them
    trace/exploration_tree.yaml # 4 lineages with the dead ends recorded verbatim
  configs/  data/  generated/  scripts/  figures/   # the evidence the claims bind to
```

Every count in `paper/ara/logic/claims.md` recomputes from `paper/data/` and `paper/generated/`
with no network access, no model calls, and no licensed data. What cannot be reproduced without
a CRSP/Compustat/WRDS subscription is an end-to-end discovery rerun; `paper/ara/PAPER.md` states
that boundary explicitly.

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[paper]'
```

Copy `.env.example` to `.env` and fill in only the credentials needed for the
providers and licensed data sources you intend to use. Never commit `.env`.

## Replication

See [`paper/README.md`](paper/README.md) for the exact figure and table
regeneration commands. The commands operate on the frozen artifacts included
in this release and do not make model or data-provider calls.

For a targeted discovery run after configuring data and a model provider:

```bash
python scripts/experiments/run_lab.py \
  --config paper/configs/baseline.yaml \
  --pair accrual_quality
```

## Repository scope

This is a clean public release with a new Git history. It excludes tests,
internal project documentation, private workflow materials, paper source, local
caches, and restricted raw data.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
