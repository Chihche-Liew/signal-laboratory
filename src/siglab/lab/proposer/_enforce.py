"""Shared proposal-count enforcement for proposer topologies (P2-4)."""
from __future__ import annotations

import sys

from siglab.lab.archive import Proposal


def enforce_proposal_count(
    proposals: list[Proposal],
    n: int,
    *,
    proposer: str,
) -> tuple[list[Proposal], list[Proposal]]:
    """Enforce the requested per-generation batch size.

    Returns ``(kept, overflow)``: ``kept`` is at most the first ``n``
    proposals in their original order; ``overflow`` is everything truncated
    off, which the caller MUST route into ``considered`` so the archive trace
    still records every generated candidate (DiscoveryLoop invariant).
    Undershoot is warned about but never padded — the LLM simply produced
    fewer than requested.
    """
    if len(proposals) > n:
        print(
            f"[{proposer}] WARNING: LLM returned {len(proposals)} proposals; "
            f"truncating to requested n={n} (overflow recorded as considered)",
            file=sys.stderr,
        )
        return proposals[:n], proposals[n:]
    if len(proposals) < n:
        print(
            f"[{proposer}] WARNING: LLM returned {len(proposals)} proposals; "
            f"{n} were requested (continuing without padding)",
            file=sys.stderr,
        )
    return proposals, []
