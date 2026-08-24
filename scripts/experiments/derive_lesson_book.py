"""Derive mechanism-level LessonBook records from completed run archives."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

from siglab.lab.lesson import ArchiveLessonBookBuilder
from siglab.lab.llm.factory import build_llm, default_model_for_provider


def _read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.resolved.json"
    if not path.exists():
        raise FileNotFoundError(f"resolved run config not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"resolved run config must be a JSON object: {path}")
    return payload


def _default_analyzer_config(run_dir: Path) -> dict[str, Any]:
    config = _read_config(run_dir)
    proposer = config.get("proposer", {})
    proposer_extra = proposer.get("extra", {}) if isinstance(proposer, dict) else {}
    role_llms = (
        proposer_extra.get("llms", {})
        if isinstance(proposer_extra, dict)
        else {}
    )
    role_config = role_llms.get("proposer", {}) if isinstance(role_llms, dict) else {}
    if isinstance(role_config, dict) and role_config.get("provider"):
        return dict(role_config)
    llm_config = config.get("llm", {})
    return dict(llm_config) if isinstance(llm_config, dict) else {}


def _analyzer_config(
    run_dir: Path,
    *,
    provider: str | None,
    model: str | None,
) -> dict[str, Any]:
    config = _default_analyzer_config(run_dir)
    if provider is not None:
        config["provider"] = provider
    if model is not None:
        config["model"] = model
    if not config.get("provider"):
        raise ValueError(
            "analyzer provider is missing; pass --provider and --model"
        )
    if not config.get("model"):
        config["model"] = default_model_for_provider(config["provider"])
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build mechanism-level lesson_book.jsonl files from completed "
            "archive.json run artifacts"
        )
    )
    parser.add_argument("run_dirs", nargs="+", help="Completed run directories")
    parser.add_argument("--provider", help="Override analyzer LLM provider")
    parser.add_argument("--model", help="Override analyzer LLM model")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    for raw_path in args.run_dirs:
        run_dir = Path(raw_path)
        llm = build_llm(
            _analyzer_config(
                run_dir,
                provider=args.provider,
                model=args.model,
            )
        )
        output = ArchiveLessonBookBuilder(llm).derive_and_write(run_dir)
        print(f"{run_dir}: wrote {output}")


if __name__ == "__main__":
    main()
