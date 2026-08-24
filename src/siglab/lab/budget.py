"""Token / wall-clock budget tracker for a DiscoveryLoop run.

Currency accounting was removed after the multi-provider migration: thinking
token attribution across Anthropic / OpenAI / Gemini can't be priced
consistently (providers bill reasoning differently and not all of it is
visible in the SDK response), so we just track tokens and wall clock.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    """Inclusive caps: a total EQUAL to its cap counts as exceeded (>=).

    Granularity: DiscoveryLoop checks exceeded() only BETWEEN generations
    (top of the while-loop, before proposing), never inside one, so the
    worst-case overshoot is one full generation of LLM calls + evaluation.
    """
    max_tokens_in: int | None
    max_tokens_out: int | None
    max_wall_clock_seconds: float | None = None

    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cached_in: int = 0
    total_cache_creation_in: int = 0
    _started_at: float = field(default_factory=time.time)

    def record_call(
        self, *,
        input_tokens: int, output_tokens: int,
        cached_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        self.total_tokens_in += input_tokens
        self.total_tokens_out += output_tokens
        self.total_cached_in += cached_input_tokens
        self.total_cache_creation_in += cache_creation_input_tokens

    def exceeded(self) -> bool:
        return bool(self.reason())

    def reason(self) -> str:
        if self.max_tokens_in is not None and self.total_tokens_in >= self.max_tokens_in:
            return f"input tokens {self.total_tokens_in} >= cap {self.max_tokens_in}"
        if self.max_tokens_out is not None and self.total_tokens_out >= self.max_tokens_out:
            return f"output tokens {self.total_tokens_out} >= cap {self.max_tokens_out}"
        if (self.max_wall_clock_seconds is not None and
                time.time() - self._started_at >= self.max_wall_clock_seconds):
            return "wall-clock exceeded"
        return ""
