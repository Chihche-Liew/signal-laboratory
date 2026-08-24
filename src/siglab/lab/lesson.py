"""Post-run mechanism beliefs derived from completed run archives."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from siglab.lab.llm.base import LLMClient


MECHANISM_STATUSES = {"promising", "mixed", "unsupported", "unresolved"}
REALIZATION_STATUSES = {"effective", "mixed", "ineffective", "unresolved"}


@dataclass(frozen=True)
class ArchiveEvidence:
    evidence_id: str
    proposal_name: str
    generation: int
    expression: str
    hypothesis: str
    reasoning: str
    expected_sign: str
    interaction_type: str
    outcome: str
    fmb_tstat: float | None
    ls_alpha: float | None
    ls_talpha: float | None
    ls_sharpe: float | None
    coverage: float | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Belief:
    status: str
    judgment: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class MechanismLesson:
    lesson_id: str
    run_id: str
    task_id: str
    mechanism_id: str
    mechanism_claim: str
    mechanism_belief: Belief
    realization_belief: Belief
    supporting_evidence: list[ArchiveEvidence] = field(default_factory=list)
    opposing_evidence: list[ArchiveEvidence] = field(default_factory=list)
    realization_failures: list[ArchiveEvidence] = field(default_factory=list)
    open_question: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "mechanism_id": self.mechanism_id,
            "mechanism_claim": self.mechanism_claim,
            "mechanism_belief": self.mechanism_belief.to_dict(),
            "realization_belief": self.realization_belief.to_dict(),
            "supporting_evidence": [
                evidence.to_dict() for evidence in self.supporting_evidence
            ],
            "opposing_evidence": [
                evidence.to_dict() for evidence in self.opposing_evidence
            ],
            "realization_failures": [
                evidence.to_dict() for evidence in self.realization_failures
            ],
            "open_question": self.open_question,
        }


@dataclass
class LessonBook:
    lessons: list[MechanismLesson] = field(default_factory=list)

    def to_jsonl_lines(self) -> list[str]:
        return [
            json.dumps(lesson.to_dict(), sort_keys=True)
            for lesson in self.lessons
        ]

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(self.to_jsonl_lines())
        output.write_text(f"{text}\n" if text else "")
        return output


class ArchiveLessonBookBuilder:
    """Use an AI synthesis pass to group archive evidence into mechanisms.

    Evidence polarity is assigned deterministically before the model sees it.
    The model may group evidence and explain beliefs, but it cannot hide or
    relabel evaluated results: every archive evidence ID must appear exactly
    once in the returned mechanism groups.
    """

    SYSTEM_PROMPT = """\
You create a compact research-judgment artifact for other AI research agents.
Group evaluated signal attempts into coherent economic mechanisms using only
their hypotheses, reasoning, expressions, and measured results.

Rules:
1. Every evidence_id must appear exactly once across all mechanism groups.
2. Do not invent mechanisms or causal claims unsupported by the supplied text.
3. Keep belief about the economic mechanism separate from belief about its
   attempted realizations.
4. Evidence outcome labels are fixed. Do not relabel or omit inconvenient
   evidence.
5. A mechanism with fewer than two usable evaluations must remain unresolved.
6. Evaluation failures bear on realization quality, not mechanism quality.
7. Supporting and opposing evidence must both be addressed in the judgments.
8. Return JSON only, with no markdown fence or commentary.
"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def derive(self, run_dir: str | Path) -> LessonBook:
        run_path = Path(run_dir)
        manifest = _read_json(run_path / "manifest.json")
        archive = _read_json(run_path / "archive.json")
        task = _read_json_if_present(run_path / "task.json")
        run_id = str(manifest["experiment_id"])
        task_id = str(manifest.get("task_id") or task.get("task_id") or "")
        threshold = float(
            task.get("evaluation_protocol", {}).get("success_threshold", 1.96)
        )
        evidence = build_archive_evidence(
            archive.get("evaluated", []),
            threshold=threshold,
        )

        if not evidence:
            return LessonBook()

        session = self.llm.start_session(system=self.SYSTEM_PROMPT)
        response = session.send(
            build_synthesis_prompt(
                run_id=run_id,
                task_id=task_id,
                evidence=evidence,
            )
        )
        synthesis = parse_synthesis_response(response.text)
        return build_lesson_book(
            run_id=run_id,
            task_id=task_id,
            evidence=evidence,
            synthesis=synthesis,
        )

    def derive_and_write(self, run_dir: str | Path) -> Path:
        run_path = Path(run_dir)
        book = self.derive(run_path)
        return book.write(run_path / "lesson_book.jsonl")


def build_archive_evidence(
    evaluated: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[ArchiveEvidence]:
    evidence = []
    for index, signal in enumerate(evaluated):
        tstat = _finite_float_or_none(signal.get("fmb_tstat"))
        error = signal.get("error")
        expected_sign = str(signal.get("expected_sign", ""))
        outcome = _evidence_outcome(
            tstat=tstat,
            error=error,
            expected_sign=expected_sign,
            threshold=threshold,
        )
        evidence.append(
            ArchiveEvidence(
                evidence_id=f"e{index:04d}",
                proposal_name=str(signal.get("name", "")),
                generation=int(signal.get("generation", 0)),
                expression=str(signal.get("expression", "")),
                hypothesis=str(signal.get("hypothesis", "")),
                reasoning=str(signal.get("reasoning", "")),
                expected_sign=expected_sign,
                interaction_type=str(signal.get("interaction_type", "")),
                outcome=outcome,
                fmb_tstat=tstat,
                ls_alpha=_finite_float_or_none(signal.get("ls_alpha")),
                ls_talpha=_finite_float_or_none(signal.get("ls_talpha")),
                ls_sharpe=_finite_float_or_none(signal.get("ls_sharpe")),
                coverage=_finite_float_or_none(signal.get("coverage")),
                error=None if error is None else str(error),
            )
        )
    return evidence


def build_synthesis_prompt(
    *,
    run_id: str,
    task_id: str,
    evidence: list[ArchiveEvidence],
) -> str:
    payload = {
        "run_id": run_id,
        "task_id": task_id,
        "outcome_definitions": {
            "supporting": "significant and in the hypothesized direction",
            "opposing_weak": "not significant under the primary metric",
            "opposing_wrong_sign": (
                "significant but opposite to the hypothesized direction"
            ),
            "realization_failure": (
                "evaluation errored or produced no finite primary metric"
            ),
        },
        "evidence": [item.to_dict() for item in evidence],
        "required_output": {
            "mechanisms": [
                {
                    "mechanism_id": "stable_snake_case_id",
                    "mechanism_claim": (
                        "concise economic claim grounded in supplied hypotheses"
                    ),
                    "evidence_ids": ["e0000"],
                    "mechanism_belief": {
                        "status": (
                            "promising | mixed | unsupported | unresolved"
                        ),
                        "judgment": "short evidence-balanced judgment",
                    },
                    "realization_belief": {
                        "status": (
                            "effective | mixed | ineffective | unresolved"
                        ),
                        "judgment": "short judgment about attempted expressions",
                    },
                    "open_question": (
                        "single question whose answer would most change the belief"
                    ),
                }
            ]
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def parse_synthesis_response(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        candidate,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced is not None:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("lesson synthesis did not return valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("mechanisms"), list):
        raise ValueError("lesson synthesis must contain a mechanisms list")
    return payload


def build_lesson_book(
    *,
    run_id: str,
    task_id: str,
    evidence: list[ArchiveEvidence],
    synthesis: dict[str, Any],
) -> LessonBook:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    used_ids: list[str] = []
    lessons = []

    for raw in synthesis["mechanisms"]:
        if not isinstance(raw, dict):
            raise ValueError("each synthesized mechanism must be an object")
        mechanism_id = _required_text(raw, "mechanism_id")
        mechanism_claim = _required_text(raw, "mechanism_claim")
        evidence_ids = raw.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError(f"mechanism {mechanism_id!r} must cite evidence_ids")
        if any(not isinstance(evidence_id, str) for evidence_id in evidence_ids):
            raise ValueError("evidence_ids must contain strings")

        unknown = set(evidence_ids) - set(evidence_by_id)
        if unknown:
            raise ValueError(f"unknown evidence IDs: {sorted(unknown)}")
        used_ids.extend(evidence_ids)
        grouped = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]

        mechanism_belief = _parse_belief(
            raw.get("mechanism_belief"),
            allowed_statuses=MECHANISM_STATUSES,
            label="mechanism_belief",
        )
        realization_belief = _parse_belief(
            raw.get("realization_belief"),
            allowed_statuses=REALIZATION_STATUSES,
            label="realization_belief",
        )
        usable = [item for item in grouped if item.outcome != "realization_failure"]
        if len(usable) < 2 and mechanism_belief.status != "unresolved":
            raise ValueError(
                f"mechanism {mechanism_id!r} has fewer than two usable "
                "evaluations and must remain unresolved"
            )

        lessons.append(
            MechanismLesson(
                lesson_id=f"{run_id}:{mechanism_id}",
                run_id=run_id,
                task_id=task_id,
                mechanism_id=mechanism_id,
                mechanism_claim=mechanism_claim,
                mechanism_belief=mechanism_belief,
                realization_belief=realization_belief,
                supporting_evidence=[
                    item for item in grouped if item.outcome == "supporting"
                ],
                opposing_evidence=[
                    item
                    for item in grouped
                    if item.outcome
                    in {"opposing_weak", "opposing_wrong_sign"}
                ],
                realization_failures=[
                    item
                    for item in grouped
                    if item.outcome == "realization_failure"
                ],
                open_question=_required_text(raw, "open_question"),
            )
        )

    expected_ids = list(evidence_by_id)
    duplicate_ids = sorted(
        evidence_id
        for evidence_id in set(used_ids)
        if used_ids.count(evidence_id) > 1
    )
    if duplicate_ids:
        raise ValueError(f"evidence IDs used more than once: {duplicate_ids}")
    missing_ids = sorted(set(expected_ids) - set(used_ids))
    if missing_ids:
        raise ValueError(f"archive evidence omitted from synthesis: {missing_ids}")
    return LessonBook(lessons=lessons)


def _evidence_outcome(
    *,
    tstat: float | None,
    error: Any,
    expected_sign: str,
    threshold: float,
) -> str:
    if error is not None or tstat is None:
        return "realization_failure"
    if abs(tstat) <= threshold:
        return "opposing_weak"
    expected_positive = expected_sign == "positive"
    actual_positive = tstat > 0
    return "supporting" if expected_positive == actual_positive else "opposing_wrong_sign"


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_belief(
    raw: Any,
    *,
    allowed_statuses: set[str],
    label: str,
) -> Belief:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    status = _required_text(raw, "status")
    if status not in allowed_statuses:
        raise ValueError(f"invalid {label} status: {status!r}")
    return Belief(status=status, judgment=_required_text(raw, "judgment"))


def _required_text(raw: dict[str, Any], field_name: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required run artifact not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"run artifact must contain a JSON object: {path}")
    return payload


def _read_json_if_present(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}
