# Problem

## Observations

### O1: Asset pricing has a hidden search denominator
The evidentiary weight of a reported return predictor depends not only on its $t$-statistic
but on how many ideas, transformations, signs, and implementations were examined before it
was selected. Published research generally reveals successful specifications, not the
unsuccessful search that produced them.

### O2: Conventional thresholds are inadequate under that denominator
Harvey, Liu and Zhu (2016) show why a $t$-statistic of 2 is not a defensible bar once
multiple testing is accounted for. Harvey (2017) documents how unreported tests, publication
incentives, and specification search weaken the credibility of financial evidence.

### O3: LLMs can already do substantial parts of financial research
Lopez-Lira and Tang (2023) and Novy-Marx and Velikov (2026) show that language models extract
return-predictive information and automate parts of the research workflow.

### O4: AI makes the denominator problem worse before it makes it better
By lowering the cost of producing and evaluating hypotheses, AI scales specification search
and enlarges the hidden denominator. Cheap hypothesis generation without a recorded
denominator is strictly worse than expensive hypothesis generation.

### O5: But an instrumented environment inverts the problem
When AI operates inside a research environment that a human froze in advance, every proposal,
revision, rejection, and empirical evaluation can be recorded. The same property that enlarges
the denominator — mechanical hypothesis generation — also makes the search population
enumerable.

## Gaps

### G1: No study combines autonomous LLM signal discovery with modern post-search inference
Prior work either automates discovery without search-adjusted inference, or applies rigorous
multiple-testing corrections to a search population that was never observed.

### G2: The search denominator is asserted, not recorded
Papers report the tests they ran. No framework makes the *complete* set of evaluated
hypotheses — including the failures — an input to the statistical procedure.

### G3: Agent reasoning is discussed anecdotally, not from the record
Claims about what an AI system "learned" are typically inferred from outputs rather than read
off a preserved trace of proposals, critiques, and revisions.

## Key Insight

Separate the two roles that a human researcher normally plays at once.

**Humans design and freeze the research environment**: the symbolic hypothesis language (66
Compustat variables, 32 composable operators), the sample, the evaluation pipeline, and the
significance protocol. **Agents search inside it**: they choose which economic ideas to pursue
but cannot alter how those ideas are tested.

This separation buys two things at once. Evaluation standards cannot be influenced by the
discovery process, because the discovery process has no write access to them. And because
every hypothesis must be expressed in the frozen language and pass through the frozen
pipeline, the search population is enumerable — so the denominator becomes an observable
quantity that enters the inference.

The design consequence that follows: **the initial significance screen carries almost no
inferential weight**. The loop is an in-sample evolutionary search over a fixed historical
panel, so a high first-pass rate is an expected property of the architecture, not evidence.
The weight is placed instead on a cumulative post-search protocol applied to the recorded
family (see [claims.md](claims.md) C05-C06).
