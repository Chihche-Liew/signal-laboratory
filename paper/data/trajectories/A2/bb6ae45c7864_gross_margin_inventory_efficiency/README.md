# A2 · Gross Margin Inventory Efficiency

- Signal ID: `bb6ae45c7864`
- Pair / generation: `profit_invest` / `gen_001`
- Origin stage: critic-triggered revision (raw_responses.jsonl row 3)
- Original experiment: `paper/data/formal/ablation_gpt_5_5/2026-08-04T17-42-26_ablation_gpt_5_5_profit_invest`

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
