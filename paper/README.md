# Paper replication

This directory contains the frozen configurations, analysis inputs, scripts,
and figure outputs associated with the paper. It does not contain the paper's
source files.

## Contents

- `ara/`: the agent-native research artifact — the paper's claims, evidence
  bindings, and exploration graph in machine-readable form. Start at
  `ara/PAPER.md`;
- `configs/`: the primary configuration and three comparison configurations;
- `data/formal/`: selected completed-run artifacts used by the result scripts;
- `data/signal_catalog.json`: the frozen complete-protocol signal catalog;
- `generated/`: adjudicated mechanism and critique labels used by Sections 5
  and 7;
- `scripts/`: all scripts that generate the paper's figures and table blocks;
- `figures/`: the released figure outputs.

## Reproduce figures and tables

From the repository root, install the package and paper dependencies:

```bash
pip install -e '.[paper]'
```

Then run the scripts in this order:

```bash
python paper/scripts/section3_generate_framework_figure.py
python paper/scripts/section4_generate_tables.py
python paper/scripts/section5_generate_results.py
python paper/scripts/section6_generate_figures.py
python paper/scripts/section6_generate_results.py
python paper/scripts/section7_generate_results.py
```

The figure scripts overwrite the corresponding PDF and PNG files in
`paper/figures/`. The table scripts create or update tagged blocks in
`paper/tables.tex`. That generated file is an intermediate replication output;
the paper source itself is outside this repository and `paper/tables.tex` is
ignored by Git.

No command above calls a language model or licensed data provider. It uses the
completed, portable result artifacts included under `paper/data/` and
`paper/generated/`.

## Full experimental rerun

The frozen run specifications are:

- `baseline.yaml` (used for three independent primary executions);
- `ablation_no_critique.yaml`;
- `ablation_gpt_5_5.yaml`;
- `ablation_gpt_5_6.yaml`.

An end-to-end rerun requires licensed CRSP/Compustat/WRDS data and credentials
for the configured model providers. The public scripts write new run artifacts
rather than modifying the frozen analysis inputs in this directory.
