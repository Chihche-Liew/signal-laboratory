"""Per-generation trace persistence for DiscoveryLoop runs."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from siglab.lab.archive import (
    SignalArchive, Proposal, EvaluatedSignal, CriticNote, ModeratorNote,
)
from siglab.lab.progress import ProgressRecorder
from siglab.lab.proposer._result import LLMCallTrace


def _diagnostic_to_dict(diagnostic):
    if hasattr(diagnostic, "to_dict"):
        return diagnostic.to_dict()
    return asdict(diagnostic)


@dataclass
class Recorder:
    output_dir: Path
    pair_name: str
    run_name: str | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress = ProgressRecorder(
            output_dir=self.output_dir,
            pair_name=self.pair_name,
            run_name=self.run_name,
        )

    def record_progress(
        self,
        event: str,
        *,
        stage: str,
        generation: int | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.progress.record(
            event,
            stage=stage,
            generation=generation,
            message=message,
            metadata=metadata,
        )

    def save_generation(
        self, *,
        generation: int,
        llm_calls: list[LLMCallTrace],
        proposals: list[Proposal], results: list[EvaluatedSignal],
        considered: list[Proposal], critic_notes: list[CriticNote],
        moderator_note: ModeratorNote | None,
    ) -> Path:
        """Persist one generation's trace.

        Generation artifacts are split into stable files so downstream review
        tools can read prompts, responses, proposals, and evaluations without
        unpacking one large trace blob.
        """
        gen_dir = self.output_dir / "generations" / f"gen_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "pair": self.pair_name,
            "generation": generation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (gen_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        with (gen_dir / "rendered_prompts.jsonl").open("w") as handle:
            for call in llm_calls:
                handle.write(json.dumps({
                    "role": call.role,
                    "round": call.round,
                    "prompt": call.prompt,
                    "cache": True,
                    "input_tokens": call.input_tokens,
                    "cached_input_tokens": call.cached_input_tokens,
                }, sort_keys=True) + "\n")
        with (gen_dir / "raw_responses.jsonl").open("w") as handle:
            for call in llm_calls:
                handle.write(json.dumps({
                    "role": call.role,
                    "round": call.round,
                    "response": call.response,
                    "thinking": call.thinking,
                    "output_tokens": call.output_tokens,
                }, sort_keys=True) + "\n")
        (gen_dir / "parsed_proposals.json").write_text(
            json.dumps([asdict(p) for p in proposals], indent=2)
        )
        (gen_dir / "considered_proposals.json").write_text(
            json.dumps([asdict(c) for c in considered], indent=2)
        )
        (gen_dir / "evaluation_results.json").write_text(
            json.dumps([asdict(r) for r in results], indent=2)
        )
        (gen_dir / "critic_notes.json").write_text(
            json.dumps([asdict(c) for c in critic_notes], indent=2)
        )
        (gen_dir / "moderator_note.json").write_text(
            json.dumps(asdict(moderator_note) if moderator_note else None, indent=2)
        )
        return gen_dir

    def save_archive(self, archive: SignalArchive) -> Path:
        payload = {
            "pair": self.pair_name,
            "evaluated": [asdict(e) for e in archive.evaluated],
            "considered": [asdict(p) for p in archive.considered],
            "critic_notes": [asdict(n) for n in archive.critic_notes],
            "moderator_decisions": [asdict(n) for n in archive.moderator_decisions],
            "parse_diagnostics": [
                _diagnostic_to_dict(diagnostic)
                for diagnostic in archive.parse_diagnostics
            ],
            "validation_diagnostics": [
                _diagnostic_to_dict(diagnostic)
                for diagnostic in archive.validation_diagnostics
            ],
        }
        out = self.output_dir / "archive.json"
        out.write_text(json.dumps(payload, indent=2))
        return out

    def save_manifest(self, manifest) -> Path:
        out = self.output_dir / "manifest.json"
        out.write_text(json.dumps(manifest.to_dict(), indent=2))
        return out

    def save_resolved_config(self, config: dict) -> Path:
        out = self.output_dir / "config.resolved.json"
        out.write_text(json.dumps(config, indent=2))
        return out

    def save_task(self, task) -> Path:
        out = self.output_dir / "task.json"
        out.write_text(json.dumps(task.to_dict(), indent=2))
        return out

    def save_validation_diagnostics(self, generation: int, diagnostics: list) -> Path:
        out = self.output_dir / "validation_diagnostics.jsonl"
        with out.open("a") as handle:
            for diagnostic in diagnostics:
                payload = diagnostic.to_dict()
                payload.setdefault("generation", generation)
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return out

    def save_parser_diagnostics(self, generation: int, diagnostics: list) -> Path:
        out = self.output_dir / "parser_diagnostics.jsonl"
        with out.open("a") as handle:
            for diagnostic in diagnostics:
                payload = diagnostic.to_dict()
                payload.setdefault("generation", generation)
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return out
