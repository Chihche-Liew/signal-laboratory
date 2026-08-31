# Claims

Nine claims (C01-C09) and five untestable assumptions (U01-U05).

Field grammar. Every claim carries, in this order: `Statement`, `Status`,
`Falsification criteria`, `Proof`, `Evidence`, `Dependencies`, `Provenance`, `Tags`, then zero
or more `Caveat (label)` entries. Numbers in `Evidence` were recomputed from the files named
there. `B1`-`B3` are the primary runs, `A1`-`A3` the comparison runs; each run is a separate
testing family.

Two conventions apply throughout and are not repeated per claim: the search is adaptive and
in-sample, so no claim here is an out-of-sample result; and A1-A3 are single executions
supporting descriptive contrasts only, never causal attribution to critique, archive, or model
identity.

---

## C01: The frozen-environment loop produces high first-pass yield
- **Statement**: Within the human-frozen hypothesis language and evaluation pipeline, the loop evaluates 1,603-1,607 hypotheses per primary run, of which 59.1%-62.3% clear a $|t_{\mathrm{FMB}}|>1.96$ screen in the sign declared before evaluation.
- **Status**: supported
- **Falsification criteria**: Recomputing sign-oriented first-pass counts from `archive.json` over all 28 theme-pair tasks yields a rate outside 59.1%-62.3% for any primary run, or an evaluated count outside 1,603-1,607.
- **Proof**: [E01]
- **Evidence**: `paper/data/formal/{baseline_r1,baseline_r2,baseline_r3}/*/archive.json`, field `evaluated[].fmb_tstat` against `evaluated[].expected_sign`. B1 949/1607 = 59.1%, B2 961/1603 = 60.0%, B3 1001/1607 = 62.3%.
- **Dependencies**: []
- **Provenance**: user
- **Tags**: discovery, first-pass, funnel
- **Caveat (inferential weight)**: A property of an in-sample evolutionary search over a fixed panel. It describes search output, not the truth of the hypotheses. C05 carries the inferential weight.

## C02: First-pass yield rises across generations
- **Statement**: Across the three primary runs the median first-pass success rate rises from 43.4% in Generation 1 to 74.3% in Generation 9, while the median of the largest sign-oriented $t_{\mathrm{FMB}}$ per generation rises from 5.5 (Generation 0) to 7.2-7.7 (Generations 6-9).
- **Status**: supported
- **Falsification criteria**: Fit an OLS trend of per-generation first-pass rate on generation index over Generations 1-9, per primary run. The claim fails if the fitted slope is not positive in at least two of the three runs; or if the cross-run median Generation-9 rate does not exceed the median Generation-1 rate by at least 10 percentage points; or if the cross-run median per-generation maximum statistic in Generations 6-9 falls below its Generation-0 value.
- **Proof**: [E02]
- **Evidence**: `paper/data/formal/{baseline_r1,baseline_r2,baseline_r3}/*/archive.json` grouped by `evaluated[].generation`; figure `paper/figures/fig1_generation_dynamics.png`.
- **Dependencies**: [C01]
- **Provenance**: user
- **Tags**: adaptation, generations
- **Caveat (Generation 0)**: Generation 0 is excluded from the rate comparison. It holds fewer evaluated hypotheses (108/106/111 against 164-168 later) and is disproportionately filtered by the cross-theme validator.
- **Caveat (attribution)**: The per-generation maximum is a within-generation extremum and depends partly on the number of evaluations, so it is read descriptively: later generations' higher success rates are not accompanied by lower maximum statistics. This claim attributes the rise to no specific component.

## C03: High univariate yield does not imply incremental conditional content
- **Statement**: The rate at which hypotheses clear the univariate screen is not sufficient to characterize search output, because it can move independently of the rate at which hypotheses retain within-task conditional content. A1 has the highest first-pass rate (75.1%, 1,211 of 1,612) yet the lowest horse-race survivor share (62, 3.85% of evaluated), below the primary-run range of 4.54%-5.10%; A3 shows the converse, a 47.5% first-pass rate with 85 survivors (5.26%), above that range. Within the three primary runs alone the two orderings are rank-identical, so the divergence is driven by the single-run comparison configurations.
- **Status**: supported
- **Falsification criteria**: Recomputed survivor shares place A1 inside or above the B1-B3 range, or place A3 below it; or the two orderings agree across all six runs.
- **Proof**: [E03]
- **Evidence**: `paper/data/formal/*/*/archive.json` and `paper/data/formal/*/*/selection/horse_race.json` field `n_survivors`. B1 73/1607 = 4.54%, B2 78/1603 = 4.87%, B3 82/1607 = 5.10%, A1 62/1612 = 3.85%, A2 69/1612 = 4.28%, A3 85/1615 = 5.26%.
- **Dependencies**: [C01]
- **Provenance**: user
- **Tags**: selection, horse-race, configuration
- **Caveat (strength)**: The claim is that the two margins can separate, not that they are statistically independent. n=6 could not support the latter.
- **Caveat (configuration)**: **Canonical statement of the A-run confound; C04, C05, and the trace refer here.** A1 removes critique and revision together, and makes fewer model calls with substantially fewer output tokens, so the contrast bundles search topology with compute and cannot separate them. Each comparison configuration was executed once. Nothing in this artifact identifies a causal effect of critique. A-run counts characterize realized output under a configuration; they must never be used to rank configurations, and never read as "less scaffolding, better output".

## C04: Critique is used frequently and coincides with broader mechanism coverage
- **Statement**: In B1-B3 the critic flags 49%-53% of requested proposal slots and triggers a revision in 839 of 840 pair-generations. Under outcome-blind mechanism coding the primary runs use 7.57-7.75 coded mechanisms per task with within-task Herfindahl indices of 0.210-0.236, whereas A1 uses 6.82 mechanisms with a Herfindahl index of 0.334.
- **Status**: supported
- **Falsification criteria**: Recounting critic verdicts against the requested-slot denominator gives a flag rate outside 49%-53% for any primary run; or the adjudicated mechanism labels, with the residual `other` code excluded, place A1's mechanism count inside the B1-B3 range or its Herfindahl index below it.
- **Proof**: [E04]
- **Evidence**: `paper/data/formal/*/*/generations/gen_*/critic_notes.json` field `verdict`; `paper/generated/mechanism_codes.csv` with `paper/generated/mechanism_coding_agreement.json`. Flags 828/886/863 on 1,680 requested slots = 49.3%/52.7%/51.4%; mechanisms per task 7.75/7.64/7.57 (B1-B3) against 6.82 (A1); HHI 0.210/0.225/0.236 against 0.334.
- **Dependencies**: [C03]
- **Provenance**: user
- **Tags**: critique, mechanism-diversity
- **Caveat (conventions)**: Both figures are slot-denominated and exclude the residual `other` code. The qualitative ordering holds under either convention; the alternative values are in E04.
- **Caveat (interpretation)**: Association, not causation. A higher Herfindahl index denotes concentration in fewer mechanisms. The coding is outcome-blind but model-produced; reliability ships in the agreement file. These observations show critique is used and coincides with broader realized coverage. They do not quantify its contribution — see C03 caveat (configuration).

## C05: Cumulative post-search validation eliminates 97.6% of horse-race records
- **Statement**: The six runs yield 449 horse-race survivor records. A seven-gate cumulative protocol reduces them $449 \to 325 \to 207 \to 103 \to 11$: 325 clear run-level Benjamini-Yekutieli FDR at $q=0.05$, 207 are not classified as declining, 103 clear direction-consistent alphas under FF5, FF6, and the HXZ $q$-factor model simultaneously, and 11 retain a direction-consistent conditional statistic above 1.96 against selected Chen-Zimmermann predictors.
- **Status**: supported
- **Falsification criteria**: Re-executing the gate intersection over the shipped post-hoc artifacts produces any stage count differing from 449 / 325 / 207 / 103 / 11, or a terminal signal set differing from the 11 records in `signal_catalog.json`.
- **Proof**: [E05]
- **Evidence**: `paper/data/formal/posthoc/<run>/{multiple_testing,double_bootstrap,subsample,multi_model_alpha,spanning}/` intersected by `paper/scripts/section6_generate_results.py::gate_rows`. Per-run BY counts 59/52/68 (B1/B2/B3), non-decay 38/37/45, all-three-factor 19/18/23, terminal 2/2/2. Comparison runs contribute 4 (A1), 1 (A2), 0 (A3).
- **Dependencies**: [C01, C03]
- **Provenance**: user
- **Tags**: multiple-testing, validation, attrition
- **Caveat (non-binding gates)**: Within the intersected population the Bayesian, signed double-bootstrap, and four-of-seven-window gates eliminate no records. This is a property of the selected population, not evidence that the procedures are equivalent or that these screens fail to bind in the full testing families.
- **Caveat (terminal counts)**: Terminal counts must not be used to rank configurations. These are small counts and each comparison configuration is observed once; A1's 4 survivors is a single-run outcome on a count of four, from a configuration differing in both search topology and compute (C03). The architecture evidence is C03 and C04, resting on the full discovery populations.
- **Caveat (family size)**: The multiplicity family is the set of hypotheses with usable full-sample statistics — 1,603 / 1,599 / 1,607 / 1,610 / 1,612 / 1,614 for B1/B2/B3/A1/A2/A3 — 0-4 records smaller per run than the evaluated counts in C01. The difference is records without usable statistics, not an inconsistency.
- **Caveat (what the procedure conditions on)**: The procedures condition on the realized set of executable hypotheses. They do not rerun the proposer-critic-revision architecture under the null, so this is multiplicity-adjusted evidence within the recorded testing family, not end-to-end selective inference for the search algorithm. The temporal gates use data the discovery loop already observed and are robustness diagnostics, not prospective out-of-sample tests. See U01.

## C06: Conditional tests against published predictors are the binding screen
- **Statement**: Among the seven gates, the conditional test against selected Chen-Zimmermann predictors is the most selective at its stage: of the 103 records entering it, 101 yield estimable regressions and 11 survive. For every estimable record the absolute conditional statistic falls below the absolute univariate statistic, and only three remain above 3.00.
- **Status**: supported
- **Falsification criteria**: Any estimable record shows an absolute conditional statistic at or above its absolute univariate statistic; or the 103 -> 11 transition is not the largest proportional reduction among the gates applied to a common input population.
- **Proof**: [E06]
- **Evidence**: `paper/data/formal/posthoc/<run>/spanning/`, fields `fmb_t_univariate`, `fmb_t_conditional`, `n_valid_months`, `direction_consistent`; figure `paper/figures/fig5_spanning.png`. 103 in, 101 estimable, 11 survive at 1.96, 3 above 3.00 (3.23/3.17/3.38), 0 of 101 violations.
- **Dependencies**: [C05]
- **Provenance**: user
- **Tags**: spanning, published-predictors, novelty
- **Caveat (comparator)**: `fmb_t_univariate` equals `fmb_t_original` for all 442 shipped spanning rows to three decimals, so the comparator is the full-sample statistic, not one re-estimated on the shortened common sample. For the four short-coverage records of C09 this is not a like-for-like comparison.
- **Caveat (scope of survival)**: Survival establishes incremental conditional content relative to the selected controls on the common estimation sample. It does not establish literature-wide novelty, and does not show that any underlying variable or economic idea is new. For how thin "selected controls" becomes on the short-coverage records, see C09.

## C07: Repeatability holds at the mechanism level, not the expression level
- **Statement**: Independent primary runs converge on discovery structure but not on outputs. Aggregate funnels are similar and pair-level first-pass ranks have cross-run Spearman correlations of 0.486-0.535. Exact recurrence is rare in the evaluated population: one to three normalized expressions reappear in any pair of primary-run evaluation sets (B1-B2 3, B1-B3 1, B2-B3 2). Among horse-race survivors recurrence is absent: eight mechanism labels recur in all three runs, but normalized-expression overlap is zero in all three run pairs.
- **Status**: supported
- **Falsification criteria**: A normalized expression is found among the horse-race survivors of two or more primary runs; or pairwise normalized-expression overlap in the full evaluation sets falls outside 1-3; or cross-run Spearman rank correlations fall outside 0.486-0.535; or all-three-run survivor label recurrence differs from eight.
- **Proof**: [E07]
- **Evidence**: `paper/data/formal/{baseline_r1,baseline_r2,baseline_r3}/*/archive.json` for evaluation-set overlap; `.../selection/horse_race.json` for survivor overlap; `paper/generated/mechanism_codes.csv` for label recurrence. Spearman 0.486/0.502/0.535; 198 of 376 labels in all three runs; 8 survivor labels in all three runs; survivor expression overlap 0/0/0.
- **Dependencies**: [C01]
- **Provenance**: user
- **Tags**: repeatability, path-dependence
- **Caveat (taxonomy granularity)**: The task-mechanism-sign taxonomy is coarse and finite and each task generates many hypotheses, so overlap at that level shows independent runs revisit similar economic claims, not that they recover the same formulas. See U02.
- **Caveat (reading)**: Expression-level non-recurrence is a reported finding, not a reproduction failure.

## C08: The recorded trace makes hypothesis revision auditable
- **Statement**: For four trajectories terminating in complete-protocol survivors, recorded feedback available at the time is followed by a specific, identifiable change in the executable expression. The changes fall under three principal dimensions — construct measurement, economic conditioning, and functional form — which `trace/exploration_tree.yaml` records at finer granularity as five `change_dimension` tags: `construct_measurement` (S1), `representational_extension` (S2), `economic_conditioning` (S4), `functional_form` (S6), `functional_role` (S10). Each link is checkable against the stored critic rationale or archived empirical result that preceded it.
- **Status**: supported
- **Falsification criteria**: For any of the four trajectories, the cited critic rationale or archived result does not exist at the stated run/pair/generation; or the subsequent expression does not differ from its predecessor in the dimension its node tags; or a node's `change_dimension` tag is absent from the five listed above.
- **Proof**: [E08]
- **Evidence**: `trace/exploration_tree.yaml`, each node carrying its `source` path into `paper/data/formal/`. Example: the S1 predecessor's flag is at `baseline_r1/2026-08-03T05-56-18_baseline_profit_invest/generations/gen_008/critic_notes.json`, target `buy_to_build_spending_ratio`.
- **Dependencies**: [C05]
- **Provenance**: user
- **Tags**: trace, revision, process
- **Caveat (selection)**: The four cases are selected ex post for survival and legibility. Every dead end shown was repaired within its own lineage. See `trace/exploration_tree.yaml` `metadata.selection_caveat` and `metadata.selection_caveat_second_order`.
- **Caveat (what a trajectory shows)**: These cases show that such changes occur and are auditable. They do not establish how often, their average value, the causal contribution of critique, or the truth of the attached economic mechanisms. The trace records observable changes in the research record, not the model's latent reasoning.

## C09: Survivor evidence is uneven and four records rest on short common samples
- **Statement**: Among the 11 complete-protocol survivors, four have Chen-Zimmermann conditional estimates based on 24-28 monthly cross sections because joint coverage with the selected controls sharply shortens the common sample; the other seven use 437-737 months. The 11 rows are run-specific records rather than necessarily distinct anomaly families — S1 and S2 are alternative expressions of the same build-versus-buy mechanism.
- **Status**: supported
- **Falsification criteria**: `n_valid_months` in the shipped spanning artifacts shows a set of short-coverage survivors other than exactly four at 24-28 months; or the remaining seven fall outside 437-737 months; or any of the four short-coverage records is shown to have been tested against the full selected control set without coverage-driven dropping.
- **Proof**: [E09]
- **Evidence**: `paper/data/signal_catalog.json`, per-record `evidence.spanning.n_valid_months` and `.skipped_controls`; `paper/data/formal/posthoc/<run>/spanning/`. Short: 24, 24, 27, 28. Long: 437, 587, 603, 636, 737, 737, 737.
- **Dependencies**: [C05, C06]
- **Provenance**: user
- **Tags**: limitations, coverage, scope
- **Caveat (control-set truncation)**: For the short-coverage records the common sample is short and the comparison set is truncated. S2 (24 months) was tested against 8 controls, with 194 Chen-Zimmermann predictors dropped as `insufficient_coverage`, among them `BM`, `Accruals`, `AssetGrowth`, `OperProf`, `RD`, and `NOA`. S4 and S9 lost 3 and 2 controls; the seven long-sample survivors lost at most one.
- **Caveat (how to quote the survivor count)**: Short-sample conditional statistics are evidence within their available common samples, not full-period spanning and not conclusive evidence of novelty. This claim bounds C05 and C06; the survivor count should not be quoted without it.

---

# Untestable assumptions

Load-bearing propositions that nothing shipped here can falsify. A verifying agent must not
report them as validated. Same field grammar; `Affects` lists the claims that rest on each.

## U01: Benjamini-Yekutieli is appropriate for an adaptive, feedback-driven search
- **Kind**: untestable-within-artifact
- **Statement**: BY controls the false discovery rate under arbitrary dependence given valid individual $p$-values. The search that produced this family was adaptive, so the $p$-values are not independent draws and the family itself was shaped by earlier outcomes.
- **Affects**: [C05, C06]
- **Testable by**: re-running the pipeline on licensed data under a null-generating process, or an argument rather than a computation.

## U02: The model-produced mechanism coding is a valid taxonomy
- **Kind**: untestable-within-artifact
- **Statement**: C04's diversity measures and C07's recurrence counts both rest on the adjudicated mechanism labels. Inter-pass reliability ships; validity as a taxonomy does not.
- **Affects**: [C04, C07]
- **Testable by**: independent human coding of a sample against the same codebook.

## U03: The within-task horse race is the right conditional stage
- **Kind**: untestable-within-artifact
- **Statement**: C03 and C05 both start from the 449 records the horse race selects. A different conditioning set would produce a different funnel.
- **Affects**: [C03, C05]
- **Testable by**: re-running selection under alternative conditioning sets on licensed data.

## U04: The seven gates are cumulative in a meaningful order
- **Kind**: untestable-within-artifact
- **Statement**: The protocol asserts that failure at one gate cannot be offset by strength at another, and applies the gates in a fixed order. Neither the ordering nor the non-compensatory structure is tested.
- **Affects**: [C05, C06]
- **Testable by**: permutation of gate order over the full testing families.

## U05: A survivor predicts returns out of sample
- **Kind**: untestable-within-artifact
- **Statement**: No claim in this artifact asserts this. The sample is the discovery sample throughout, and the temporal gates use data the loop already observed.
- **Affects**: [C05, C06, C09]
- **Testable by**: freezing the expressions and protocol and evaluating prospectively on newly realized returns.

---

Scope of the falsification criteria above: C01, C05, C07's Spearman clause, and C09 recompute
over frozen files shipped in this repository. They test transcription fidelity, not the
underlying result. The reproduction boundary is in [PAPER.md](../PAPER.md).
