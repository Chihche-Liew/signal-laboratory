# A1 · NegativeWorkingCapitalProfitSort

- Signal ID: `6c3d5f2d7758`
- Pair / generation: `quality_distress` / `gen_001`
- Origin stage: single-agent proposal (raw_responses.jsonl row 1)
- Original experiment: `paper/data/formal/ablation_no_critique/2026-08-03T22-42-21_ablation_no_critique_quality_distress`

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
