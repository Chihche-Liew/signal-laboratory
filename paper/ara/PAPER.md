---
title: "Can AI Do Financial Research? LLM-Guided Hypothesis Discovery in Asset Pricing"
authors: [Joyce Liu, Miao Liu, Zhizhe Liu, Danqing Mei]
year: 2026
venue: "Working paper"
ara_version: "1.0"
ara_seal_target: 2
domain: "Empirical Asset Pricing / AI for Science"
keywords: [asset pricing, LLM agent, hypothesis discovery, multiple testing, search denominator]
paper_source: not included
reproduction_boundary: "#reproduction-boundary"
claims_summary:
  C01: "The loop evaluates 1,603-1,607 hypotheses per primary run; 59.1%-62.3% clear a signed significance screen"
  C02: "First-pass yield rises across generations, median 43.4% (Gen 1) to 74.3% (Gen 9)"
  C03: "Univariate yield is not sufficient to characterize search output; it can move independently of conditional yield"
  C04: "Critique is used on about half of proposal slots and coincides with broader mechanism coverage"
  C05: "A seven-gate cumulative protocol reduces 449 horse-race records to 11"
  C06: "The conditional test against published predictors is the binding screen: 103 in, 11 out"
  C07: "Repeatability holds at the mechanism level; survivor expression overlap across runs is zero"
  C08: "The recorded trace makes individual hypothesis revisions auditable against prior feedback"
  C09: "LIMITATION: 4 of the 11 survivors rest on 24-28 month common samples against truncated control sets"
untestable_assumptions: [U01, U02, U03, U04, U05]
layers:
  logic: logic/
  src: src/
  trace: trace/
  evidence: evidence/
run_codes:
  B1: baseline_r1
  B2: baseline_r2
  B3: baseline_r3
  A1: ablation_no_critique
  A2: ablation_gpt_5_5
  A3: ablation_gpt_5_6_sol
provenance:
  "logic/*.md": user
  "trace/exploration_tree.yaml": mixed; see its metadata.verbatim_fields
  "paper/data/**": ai-executed
  "paper/generated/**": ai-suggested
abstract: >
  Specialized LLM agents run the hypothesis-discovery loop in empirical asset pricing inside a
  human-frozen environment: a symbolic language of interpretable accounting formulas and a
  fixed empirical evaluation pipeline. Across six runs the system evaluates 9,656 hypotheses.
  Because the search is adaptive and in-sample, the inferential weight rests on a cumulative
  post-search protocol rather than the initial screen.
---

# Can AI Do Financial Research?

The agent-native record of the study, and a companion to the paper rather than a replacement.
Claims, evidence bindings, and the exploration graph in machine-readable form.

## Layer index

### Cognitive layer (`/logic`)
| File | Description |
|---|---|
| [problem.md](logic/problem.md) | Observations O1-O5, gaps G1-G3, the design insight |
| [claims.md](logic/claims.md) | 9 claims (C01-C09) bound to shipped evidence, plus 5 untestable assumptions (U01-U05) |
| [experiments.md](logic/experiments.md) | 9 experiment specifications (E01-E09) |

### Exploration graph (`/trace`)
| File | Description |
|---|---|
| [exploration_tree.yaml](trace/exploration_tree.yaml) | Research DAG: 4 lineages, 21 nodes. Doubly selected; see `metadata.selection_caveat` and `metadata.selection_caveat_second_order` in the file. |

### Physical layer (`/src`)
| File | Description |
|---|---|
| [src/environment.md](src/environment.md) | Runtime, dependencies, credentials, data licensing, why reruns do not match |
| [src/artifacts.md](src/artifacts.md) | Pointer index to the codebase — the code is not copied into this artifact |

The code lives at the repository root and is indexed, not duplicated: `src/siglab/` (discovery
loop, symbolic grammar, evaluation pipeline), `scripts/experiments/run_lab.py` (run entry
point), `paper/configs/*.yaml` (the four frozen configurations).

### Evidence layer (`/evidence`)
| File | Description |
|---|---|
| [evidence/README.md](evidence/README.md) | Index binding every claim to the shipped run artifacts that ground it |

The evidence is ~105 MB of run artifacts shipped beside this artifact rather than inside it:
`paper/data/formal/`, `paper/data/signal_catalog.json`, `paper/generated/`, `paper/figures/`.

## Reproduction boundary

| Layer | Status |
|---|---|
| Claim / evidence consistency | Verifiable now. All counts recompute from `paper/data/` and `paper/generated/` with no network, no model calls, no licensed data. |
| Figure and table regeneration | Verifiable now. `paper/scripts/*.py` recompute every reported number from the frozen artifacts; run them in the order given in `paper/README.md`. |
| End-to-end discovery rerun | Not verifiable. Requires licensed CRSP/Compustat/WRDS data and model-provider credentials. |

**ARA Seal Level 3 (sandboxed execution reproducibility) is not attainable.** The pipeline reads
licensed data that cannot be redistributed. A verification agent can check every claim against
shipped evidence but cannot re-derive the underlying returns. This is an audit record, not a
replay.

LLM search is also stochastic and path dependent. Even with data access, rerunning
`baseline.yaml` produces a different search path and different final hypotheses. Expression-level
non-recurrence is a reported finding (C07), not a reproduction failure.

Each run is a separate testing family; totals across runs are descriptive. A-run counts
characterize realized output under a configuration and must not be used to rank configurations
— canonical statement in C03 `Caveat (configuration)`.
