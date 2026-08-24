"""LLMClient / LLMSession interfaces and shared types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    thinking: str = ""
    raw_content: list = field(default_factory=list)  # provider-native assistant blocks (Anthropic only)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0          # cache_read (Anthropic) / cached_tokens (OpenAI) / cached_content_token_count (Gemini)
    cache_creation_input_tokens: int = 0  # Anthropic-only; 0 on OpenAI / Gemini


class LLMSession(Protocol):
    """A multi-turn conversation bound to one system prompt + one provider.

    Encapsulates provider-specific multi-turn state transport:
      - Anthropic: local accumulator of raw content blocks (encrypted
        extended-thinking signature must survive turns).
      - OpenAI:    `previous_response_id` for server-side history; per-turn
        `instructions` are still sent explicitly.
      - Gemini:    `previous_interaction_id` for server-side history; per-turn
        `system_instruction` / `generation_config` are still sent explicitly.

    Callers only see `.send()`. The `cache` kwarg is per-turn — Anthropic
    attaches `cache_control: {type: ephemeral}` to the user message;
    OpenAI / Gemini ignore it (they do automatic prefix caching).
    """
    def send(self, user_message: str, *, cache: bool = False) -> LLMResponse: ...


class LLMClient(Protocol):
    """Session factory. Multi-turn work goes through `start_session`;
    one-shot calls create a throwaway session and send once.
    """
    def start_session(self, *, system: str) -> LLMSession: ...
