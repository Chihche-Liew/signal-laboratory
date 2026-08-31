# B2 · PredictableAccrualsSustainedProfit

- Signal ID: `50aee7f84a2b`
- Pair / generation: `accrual_quality` / `gen_007`
- Origin stage: initial proposal accepted by critic (raw_responses.jsonl row 1)
- Original experiment: `paper/data/formal/baseline_r2/2026-08-05T21-09-11_baseline_accrual_quality`

## Contents

- `generation/rendered_prompts.jsonl`: prompts sent in the producing generation.
- `generation/raw_responses.jsonl`: visible proposer/critic/revision responses and provider-returned thinking summaries when available.
- `generation/parsed_proposals.json`: final parsed batch that entered validation.
- `generation/evaluation_results.json`: in-loop evaluations for the full batch.
- `generation/critic_notes.json`: structured critic verdicts (empty for A1).
- `experiment_context/`: exact run manifest, resolved config, and task contract.
- `selection/horse_race.json`: within-pair selection artifact.
- `posthoc_evidence.json`: signal-specific extracts from every formal posthoc stage.

The generation files contain the full proposal batch, not only this signal. This preserves the contemporaneous comparison set and the actual critic/revision context.
