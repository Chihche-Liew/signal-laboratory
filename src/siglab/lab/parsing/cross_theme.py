from __future__ import annotations

import re

from siglab.lab.archive import Proposal
from siglab.lab.parsing.base import ParseDiagnostic, ParsedProposalBatch
from siglab.lab.task import DiscoveryTask


class CrossThemeProposalParser:
    def parse(
        self, text: str, *, task: DiscoveryTask, generation: int,
    ) -> ParsedProposalBatch:
        parser_text = _normalize_signal_header(text)
        proposals = [
            _parse_proposal_block(block, task=task)
            for block in _signal_blocks(parser_text)
        ]
        proposals = [proposal for proposal in proposals if proposal is not None]
        diagnostics = [_diagnostic_for_parse(proposals, task=task, generation=generation)]
        return ParsedProposalBatch(
            proposals=proposals,
            diagnostics=diagnostics,
            raw_text=text,
        )


def _normalize_signal_header(text: str) -> str:
    return re.sub(
        r"(?im)^\s*SIGNAL\s+(\d+)\s*:\s*(.+?)\s*$",
        r"--- SIGNAL \1 ---\nNAME: \2",
        text,
    )


def _signal_blocks(text: str) -> list[str]:
    blocks = re.split(r"---+\s*SIGNAL\s+\d+\s*---+", text, flags=re.IGNORECASE)
    if len(blocks) > 1:
        return blocks[1:]
    if re.search(r"(?:^|\n)\s*NAME\s*:", text, re.IGNORECASE) and re.search(
        r"(?:^|\n)\s*EXPRESSION\s*:", text, re.IGNORECASE,
    ):
        return [text]
    return []


def _parse_proposal_block(block: str, *, task: DiscoveryTask) -> Proposal | None:
    fields = _parse_fields(block)
    if "name" not in fields or "expression" not in fields:
        return None

    expected_sign = _normalize_expected_sign(fields.get("expected_sign", ""))
    interaction_type = fields.get("interaction_type", "").lower().strip()

    return Proposal(
        name=fields["name"],
        expression=fields["expression"],
        hypothesis=fields.get("hypothesis", ""),
        expected_sign=expected_sign,
        reasoning=fields.get("reasoning", ""),
        interaction_type=interaction_type,
        theme_a=str(task.metadata["theme_a"]),
        theme_b=str(task.metadata["theme_b"]),
        author="proposer",
    )


def _normalize_expected_sign(raw_value: str) -> str:
    expected_sign = raw_value.lower().strip()
    if expected_sign in {"+", "pos"}:
        return "positive"
    if expected_sign in {"-", "neg"}:
        return "negative"
    return expected_sign


def _parse_fields(block: str) -> dict[str, str]:
    field_names = [
        "NAME",
        "EXPRESSION",
        "HYPOTHESIS",
        "EXPECTED_SIGN",
        "INTERACTION_TYPE",
        "REASONING",
    ]
    fields: dict[str, str] = {}
    for field_name in field_names:
        pattern = (
            rf"(?:^|\n)\s*{field_name}\s*:\s*(.*?)"
            rf"(?=\n\s*(?:{'|'.join(field_names)})\s*:|$)"
        )
        match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        if match:
            fields[field_name.lower()] = match.group(1).strip()
    return fields


def _diagnostic_for_parse(
    proposals: list[Proposal], *, task: DiscoveryTask, generation: int,
) -> ParseDiagnostic:
    metadata = {"task_id": task.task_id}
    if proposals:
        count = len(proposals)
        return ParseDiagnostic(
            severity="info",
            code="parsed_proposals",
            message=f"parsed {count} proposals",
            generation=generation,
            metadata=metadata,
        )

    return ParseDiagnostic(
        severity="error",
        code="no_proposals_parsed",
        message="parser returned zero proposals from model output",
        generation=generation,
        metadata=metadata,
    )
