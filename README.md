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
  tables in `paper/`.

Paper source files are intentionally not part of this repository. Licensed
CRSP, Compustat, and WRDS source data are also excluded. The included completed
run artifacts are sufficient to regenerate the paper's reported figures and
tables; rerunning discovery or empirical tests from raw data requires the
corresponding data subscriptions and model credentials.

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
