"""Run-level manifest metadata for lab artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    experiment_id: str
    run_name: str
    created_at: str
    git_sha: str
    provider: str
    model: str
    task_id: str
    output_dir: str
    mode: dict
    reference_library: dict
    cache_semantics: dict
    posthoc: dict

    def to_dict(self) -> dict:
        return asdict(self)
