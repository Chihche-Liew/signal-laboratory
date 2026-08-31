# Experiments

Each experiment states what it verifies, the inputs that ship in this repository, the
procedure, and the expected outcome. All nine run without network access, model calls, or
licensed data.

Common setup unless stated otherwise: six runs (`B1`=`baseline_r1`, `B2`=`baseline_r2`,
`B3`=`baseline_r3`, `A1`=`ablation_no_critique`, `A2`=`ablation_gpt_5_5`,
`A3`=`ablation_gpt_5_6_sol`), 28 theme-pair tasks each, 10 generations, 6 requested proposals
per generation. Sample: CRSP-Compustat 1963-2024, financial firms and microcaps excluded,
NYSE breakpoints, value weights. Frozen in `paper/configs/`.

---

## E01: Discovery funnel and first-pass yield
- **Verifies**: [C01]
- **Evidence**: [evidence/README.md -> Primary run artifacts](../evidence/README.md)
- **Run**: `scripts/experiments/run_lab.py` with `paper/configs/baseline.yaml`; recomputed by `paper/scripts/section5_generate_results.py` (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: `paper/data/formal/{baseline_r1,baseline_r2,baseline_r3}/*/archive.json`
- **Procedure**: For each run, concatenate `evaluated[]` across all 28 pair directories. Count records with a non-null `fmb_tstat`. A record is first-pass when `abs(fmb_tstat) > 1.96` and the sign of `fmb_tstat` matches `expected_sign`. The denominator is all records submitted to empirical evaluation, including execution errors.
- **Metrics**: evaluated count; first-pass count; first-pass rate
- **Baselines**: The three primary runs are compared against each other; A1-A3 provide contrast in E03.
- **Expected outcome**: 1,603-1,607 evaluated; 949-1,001 first-pass; rate 59.1%-62.3%.
- **Dependencies**: []

## E02: Within-run adaptation across generations
- **Verifies**: [C02]
- **Evidence**: [evidence/README.md -> Primary run artifacts](../evidence/README.md)
- **Run**: `scripts/experiments/run_lab.py` with `paper/configs/baseline.yaml`; recomputed by `paper/scripts/section5_generate_results.py` (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: same archives as E01, plus `paper/scripts/section5_generate_results.py`
- **Procedure**: Group `evaluated[]` by `generation`. Compute the first-pass rate per generation per run, then the cross-run median. Separately compute, per generation, the largest sign-oriented `fmb_tstat`, then the cross-run median. Centre the comparison on Generations 1-9; report Generation 0 separately.
- **Metrics**: median first-pass rate by generation; median per-generation maximum sign-oriented $t$
- **Expected outcome**: rate rises 43.4% (Gen 1) -> 74.3% (Gen 9); maximum statistic 5.5 (Gen 0) -> 7.2-7.7 (Gens 6-9).
- **Dependencies**: [E01]

## E03: Univariate yield versus within-task conditional yield
- **Verifies**: [C03]
- **Evidence**: [evidence/README.md -> Primary run artifacts](../evidence/README.md)
- **Run**: `scripts/experiments/run_lab.py` across all four configurations in `paper/configs/`; recomputed by `paper/scripts/section5_generate_results.py` (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: all six runs' `archive.json` and `selection/horse_race.json`
- **Procedure**: For each run compute the first-pass rate (E01 procedure) and the horse-race survivor share, defined as the sum of `n_survivors` over the 28 pair directories divided by the evaluated count. Plot one against the other.
- **Metrics**: first-pass rate; survivor share
- **Baselines**: B1-B3 form the reference range on both dimensions.
- **Expected outcome**: A1 has the highest first-pass rate (75.1%) and the lowest survivor share (3.85%), below the B1-B3 range 4.54%-5.10%; A3 has a 47.5% first-pass rate and a 5.26% survivor share, above that range.
- **Dependencies**: [E01]

## E04: Critic usage and mechanism diversity
- **Verifies**: [C04]
- **Evidence**: [evidence/README.md -> Adjudicated labels](../evidence/README.md)
- **Run**: `scripts/experiments/run_lab.py` for the critic notes; label files recomputed by `paper/scripts/section7_generate_results.py` (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: `paper/data/formal/*/*/generations/gen_*/critic_notes.json`; `paper/generated/mechanism_codes.csv`; `paper/generated/mechanism_coding_agreement.json`
- **Procedure**: Count `verdict == "flag"` against requested proposal slots per run. Count pair-generations in which at least one revision round follows a flag. For mechanism diversity, take the adjudicated labels and compute, per task, the number of distinct coded mechanisms and the Herfindahl index of their shares; average over the 28 tasks.
- **Metrics**: flag rate; revision-triggering pair-generations; mechanisms per task; within-task Herfindahl index
- **Expected outcome**: flag rate 49%-53% in B1-B3; revisions in 839 of 840 pair-generations; mechanisms per task 7.57-7.75 (B1-B3) versus 6.82 (A1); Herfindahl 0.210-0.236 (B1-B3) versus 0.334 (A1).
- **Dependencies**: [E03]
- **Caveat (conventions)**: Canonical statement; C04 refers here. Two conventions determine the expected values. (i) The flag-rate denominator is requested proposal slots (6 x 10 x 28 = 1,680 per run), not evaluated hypotheses; on the evaluated denominator the rates are 51.5%/55.3%/53.7% and B2/B3 fall outside the stated band. (ii) The mechanism-count and Herfindahl figures exclude the residual `other` label (182 of 9,656 rows); including it gives 8.21/8.04/8.04 and HHI 0.205/0.221/0.229 for B1-B3, and 7.29 with HHI 0.323 for A1. The qualitative ordering holds either way.
- **Note**: The coding is outcome-blind — the sessions see neither the target proposal's empirical outcome nor any post-search result — but is model-produced. Reliability statistics ship alongside the labels. See U02.

## E05: Cumulative post-search gate intersection
- **Verifies**: [C05]
- **Evidence**: [evidence/README.md -> Primary run artifacts](../evidence/README.md)
- **Run**: `scripts/experiments/run_posthoc.py` and the per-gate entry points in `scripts/experiments/`; recomputed by `paper/scripts/section6_generate_results.py` (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: `paper/data/formal/posthoc/<run>/{multiple_testing,double_bootstrap,subsample,multi_model_alpha,spanning}/`; `paper/data/signal_catalog.json`
- **Procedure**: Run `paper/scripts/section6_generate_results.py`. Within each run, compute multiplicity statistics over the full family of hypotheses with usable Fama-MacBeth statistics, then intersect those decisions with that run's horse-race records. Apply the seven gates cumulatively in order: (1) BY FDR at $q=0.05$; (2) posterior null probability below 5% under the moderate prior; (3) signed double-bootstrap threshold; (4) declared-direction significance in at least four of seven fixed windows; (5) non-decay under the rolling 60-month procedure; (6) direction-consistent $t_\alpha > 1.96$ under FF5, FF6, and HXZ $q$ simultaneously; (7) direction-consistent conditional statistic above 1.96 against up to ten selected Chen-Zimmermann predictors. Failure at one gate cannot be offset elsewhere.
- **Metrics**: surviving record count after each gate, per run and pooled
- **Expected outcome**: pooled 449 -> 325 -> 207 -> 103 -> 11; per-run BY counts 59/52/68 for B1/B2/B3; terminal 2/2/2 for B1-B3 and 4/1/0 for A1/A2/A3; terminal signal ids equal to the 11 in `signal_catalog.json`.
- **Dependencies**: [E01, E03]
- **Caveat (execution)**: The script appends a tagged block to `paper/tables.tex`, a generated file that is gitignored and not shipped. Running the documented sequence in `paper/README.md` creates it at the `section4` step. To reproduce only the gate counts without writing anything, import the module and call `gate_rows()` directly.
- **Note**: The script asserts both `len(catalog) == 11` and set equality between the computed terminal ids and the catalog ids, so a mismatch fails loudly rather than silently.

## E06: Conditional tests against published predictors
- **Verifies**: [C06]
- **Evidence**: [evidence/README.md -> Primary run artifacts](../evidence/README.md)
- **Run**: `scripts/experiments/run_spanning_test.py`; recomputed by `paper/scripts/section6_generate_results.py` (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: `paper/data/formal/posthoc/<run>/spanning/`
- **Procedure**: For the 103 records entering the final gate, read `fmb_t_univariate` and `fmb_t_conditional` on the common estimation sample. Count estimable regressions, direction-consistent survivors above 1.96, and survivors above 3.00. Compare each record's absolute conditional statistic with its absolute univariate statistic.
- **Metrics**: estimable count; survivor count at 1.96 and at 3.00; sign of (|conditional| - |univariate|)
- **Expected outcome**: 101 of 103 estimable; 11 survive at 1.96; 3 above 3.00; |conditional| < |univariate| for every estimable record.
- **Dependencies**: [E05]

- **Caveat (comparator)**: `fmb_t_univariate` equals `fmb_t_original` in all 442 shipped spanning rows to three decimals, so the comparator is the full-sample statistic, not one re-estimated on the shortened common sample (C06). Record `n_controls` and `skipped_controls` as well; the control-set truncation on the short-coverage records is quantified in C09.
## E07: Cross-run repeatability at expression and mechanism level
- **Verifies**: [C07]
- **Evidence**: [evidence/README.md -> Adjudicated labels](../evidence/README.md)
- **Run**: `scripts/experiments/run_lab.py` with `paper/configs/baseline.yaml` x3; recomputed by `paper/scripts/section5_generate_results.py` (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: primary-run `selection/horse_race.json`; `paper/generated/mechanism_codes.csv`
- **Procedure**: Normalize expressions and intersect them pairwise across B1, B2, B3 at **two** population levels, keeping them separate: (a) the full evaluation sets from `archive.json`, and (b) the horse-race survivor sets from `selection/horse_race.json`. Separately intersect the sets of accounting variables used. Compute Spearman correlations of pair-level first-pass ranks across run pairs. At the mechanism level, count task-mechanism-sign labels appearing in three, two, and one run, over all evaluated hypotheses and again restricted to horse-race survivors.
- **Metrics**: normalized-expression recurrence at each population level; verbatim survivor recurrence; shared variable sets; Spearman rank correlation; label recurrence counts
- **Expected outcome**: over the **evaluation sets**, 1-3 normalized expressions per run pair (B1-B2 3, B1-B3 1, B2-B3 2). Over the **survivor sets**, **zero** overlap in all three run pairs, normalized and verbatim alike. 128-149 shared variable sets. Spearman 0.486-0.535. 198 labels in all three runs over all hypotheses, but only 8 among survivors.
- **Dependencies**: [E01]
- **Caveat (population level)**: Evaluation-set overlap is 1-3; survivor-set overlap is 0. Comparing one population's result against the other's expected value produces a spurious mismatch.

## E08: Trajectory audit of four survivor lineages
- **Verifies**: [C08]
- **Evidence**: [evidence/README.md -> Primary run artifacts](../evidence/README.md)
- **Run**: `scripts/experiments/run_lab.py`; the trace is read directly, no script (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: `trace/exploration_tree.yaml` and the `source` path recorded on each node
- **Procedure**: For each of the four trajectories, open the cited `critic_notes.json` or `archive.json` at the stated run, pair, and generation. Confirm the quoted rationale exists verbatim at that location and precedes the revision. Then diff the predecessor and successor expressions and confirm the change falls in the dimension the node claims (construct measurement, economic conditioning, or functional role).
- **Metrics**: node-to-source resolution rate; predecessor/successor expression diff; `change_dimension` tag membership in the five listed in C08
- **Expected outcome**: every node resolves; each successor differs from its predecessor in the stated dimension.
- **Dependencies**: [E05]
- **Caveat (tag granularity)**: The trace records five `change_dimension` values, finer-grained than the three principal dimensions. N07 carries an explicit `predecessor: N06` so the S1->S2 representational extension can be diffed; the other successors diff against the flagged proposal named in the preceding `dead_end` node.
- **Note**: This experiment audits the record, not the mechanisms. A passing result shows the trace is faithful, not that the economic reasoning is correct.

## E09: Survivor evidence coverage
- **Verifies**: [C09]
- **Evidence**: [evidence/README.md -> Primary run artifacts](../evidence/README.md)
- **Run**: `scripts/experiments/run_spanning_test.py`; recomputed by `paper/scripts/section6_generate_results.py` (components indexed in [src/artifacts.md](../src/artifacts.md))
- **Setup**: `paper/data/signal_catalog.json`, per-record `evidence.spanning`
- **Procedure**: Read `n_valid_months` for each of the 11 survivors. Partition into short-sample (<30 months) and long-sample records. Separately, compare the S1 and S2 expressions and hypotheses to confirm they represent one mechanism through two financial-statement contrasts.
- **Metrics**: `n_valid_months` distribution; count below 30 months
- **Expected outcome**: exactly four records with 24-28 valid months; the remaining seven with 437-737; S1 and S2 sharing the build-versus-buy mechanism.
- **Dependencies**: [E05, E06]
