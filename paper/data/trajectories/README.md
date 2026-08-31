# Per-survivor trajectory bundles

One directory per complete-protocol survivor, keyed `<code>/<signal_id>_<name>`, where `<code>`
is the run (`B1`-`B3` primary, `A1`-`A3` comparison). Eleven bundles, one for each record in
`paper/data/signal_catalog.json`; each record's `trajectory_rel` field points here.

These bundles are a **convenience view**, not a separate source of truth. Everything in them is
extracted from the full run artifacts under `paper/data/formal/`, which remain authoritative.
They exist so that a reader interested in one surviving signal does not have to reassemble its
context from six runs and 168 pair directories.

## Layout

```
<code>/<signal_id>_<name>/
  README.md                             # what this signal is and where it came from
  generation/
    rendered_prompts.jsonl              # prompts actually sent in the producing generation
    raw_responses.jsonl                 # proposer / critic / revision responses verbatim
    parsed_proposals.json               # the parsed batch that entered validation
    considered_proposals.json           # the contemporaneous comparison set
    critic_notes.json                   # structured critic verdicts (empty for A1)
    evaluation_results.json             # in-loop evaluations for the full batch
    moderator_note.json
    metadata.json
  experiment_context/
    manifest.json                       # run manifest, including the git sha
    config.resolved.json                # the fully resolved configuration
    task.json                           # the task contract
  selection/horse_race.json             # within-pair selection artifact
  posthoc_evidence.json                 # extracts from every formal post-hoc stage
```

The `generation/` files hold the **full proposal batch**, not only the surviving signal. This is
deliberate: it preserves the contemporaneous comparison set and the actual critic and revision
context in which the signal was produced.

`rendered_prompts.jsonl` contains the complete prompts as sent, including system prompts.

## Path rewriting

Paths inside these files were rewritten from the original private run layout
(`data/experiments/formal/...`, in places with Windows separators, and in eleven
`horse_race.json` files as machine-absolute paths) to this release's layout
(`paper/data/formal/...`). Every pointer resolves as shipped. The `git_sha` in each
`manifest.json` is unchanged and identifies the code revision that produced the run.

## Relation to the research artifact

`paper/ara/trace/exploration_tree.yaml` traces four of these lineages in detail, including the
dead ends that preceded them, and quotes the critic rationales verbatim from the same
`critic_notes.json` files indexed here. That trace is doubly selected — see its
`metadata.selection_caveat_second_order` — and is not a sample of how the search fails.
