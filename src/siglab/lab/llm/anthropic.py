"""Anthropic SDK wrapper: AnthropicClient + AnthropicSession."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from siglab.lab.llm.base import LLMResponse
from siglab.lab.llm.retry import call_with_retries


def _ephemeral(text: str) -> list[dict]:
    """User-message content with an ephemeral cache marker."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _uncached_content(text: str) -> list[dict]:
    """User-message content retained across turns without cache markers."""
    return [{"type": "text", "text": text}]


def _parse_anthropic_response(raw) -> LLMResponse:
    text = ""
    thinking = ""
    for block in raw.content:
        if block.type == "thinking":
            thinking = getattr(block, "thinking", "")
        elif block.type == "text":
            text = block.text
    u = raw.usage
    return LLMResponse(
        text=text,
        thinking=thinking,
        raw_content=raw.content,
        input_tokens=getattr(u, "input_tokens", 0),
        output_tokens=getattr(u, "output_tokens", 0),
        cached_input_tokens=getattr(u, "cache_read_input_tokens", 0),
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0),
    )


@dataclass
class AnthropicSession:
    """Multi-turn Anthropic conversation.

    Anthropic holds no server-side state, so the session locally accumulates
    the raw `assistant` content blocks from each turn — the encrypted
    extended-thinking signature must ride along or the API rejects the next
    turn.
    """
    _sdk_client: Any
    _model: str
    _max_tokens: int
    _thinking_type: str
    _thinking_budget: int
    _output_effort: str | None
    _temperature: float
    _system: str
    _history: list = field(default_factory=list)  # [{role, content}, ...]

    def send(self, user_message: str, *, cache: bool = False) -> LLMResponse:
        content: str | list = _ephemeral(user_message) if cache else user_message
        api_messages = list(self._history) + [{"role": "user", "content": content}]

        thinking_cfg: dict = {"type": self._thinking_type}
        if self._thinking_type == "enabled":
            thinking_cfg["budget_tokens"] = self._thinking_budget

        kwargs = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            thinking=thinking_cfg,
            messages=api_messages,
        )
        if self._output_effort:
            kwargs["output_config"] = {"effort": self._output_effort}
        if self._system:
            kwargs["system"] = self._system

        raw = call_with_retries(self._sdk_client.messages.create, kwargs=kwargs)
        resp = _parse_anthropic_response(raw)

        history_content = _uncached_content(user_message) if cache else user_message
        self._history.append({"role": "user", "content": history_content})
        self._history.append({"role": "assistant", "content": raw.content})
        return resp


@dataclass
class AnthropicClient:
    """Wraps anthropic.Anthropic. `_sdk_client` is a test-injection hook;
    in production it is left None and the wrapper instantiates the real SDK
    using ANTHROPIC_API_KEY.
    """
    model: str
    max_tokens: int = 16000
    thinking_budget: int = 10000
    thinking_type: str = "adaptive"
    output_effort: str | None = None
    temperature: float = 1.0
    _sdk_client: Any = None

    def __post_init__(self):
        if self._sdk_client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._sdk_client = anthropic.Anthropic(api_key=api_key)

    def start_session(self, *, system: str) -> AnthropicSession:
        return AnthropicSession(
            _sdk_client=self._sdk_client,
            _model=self.model,
            _max_tokens=self.max_tokens,
            _thinking_type=self.thinking_type,
            _thinking_budget=self.thinking_budget,
            _output_effort=self.output_effort,
            _temperature=self.temperature,
            _system=system,
        )
