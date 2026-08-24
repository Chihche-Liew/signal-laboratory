"""Google GenAI SDK wrapper: GeminiClient + GeminiSession (Interactions API).

Multi-turn state is held server-side via `previous_interaction_id`; the
session stores only the last interaction id locally. System prompt is passed
via `system_instruction=` on every turn because interaction-scoped parameters
are not carried forward by `previous_interaction_id`. Thinking is enabled via
`generation_config={"thinking_level": ..., "thinking_summaries": "auto"}` on
every turn.

The Interactions response returns a `steps` timeline:
  - `ThoughtStep` (type='thought'): `.summary` carries the reasoning summary
  - `ModelOutputStep` (type='model_output'): `.content` carries visible text
SDKs also expose `interaction.output_text` as the recommended text shortcut.
Usage is on `r.usage` with separate counts for input / output / thought /
cached tokens. We roll `total_thought_tokens` into `output_tokens` to match
the Anthropic / OpenAI convention where reasoning tokens are billed as output.

Caching: Gemini auto-caches long shared prefixes on pro models. The `cache`
kwarg on `session.send()` is a no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from siglab.lab.llm.base import LLMResponse
from siglab.lab.llm.retry import call_with_retries


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text_items(items) -> list[str]:
    chunks: list[str] = []
    for item in items or []:
        if _get(item, "type") == "text":
            text = _get(item, "text")
            if text:
                chunks.append(text)
    return chunks


def _parse_gemini_response(raw) -> LLMResponse:
    text = getattr(raw, "output_text", "") or ""
    thought_chunks: list[str] = []

    for step in getattr(raw, "steps", []) or []:
        step_type = _get(step, "type")
        if step_type == "thought":
            thought_chunks.extend(_text_items(_get(step, "summary", [])))
        elif step_type == "model_output" and not text:
            text = "\n".join(_text_items(_get(step, "content", [])))

    u = getattr(raw, "usage", None)
    in_tok = getattr(u, "total_input_tokens", 0) or 0 if u else 0
    out_tok = getattr(u, "total_output_tokens", 0) or 0 if u else 0
    thought_tok = getattr(u, "total_thought_tokens", 0) or 0 if u else 0
    cached = getattr(u, "total_cached_tokens", 0) or 0 if u else 0

    return LLMResponse(
        text=text,
        thinking="\n\n".join(thought_chunks),
        raw_content=[],
        input_tokens=max(0, in_tok - cached),
        output_tokens=out_tok + thought_tok,  # roll reasoning into output
        cached_input_tokens=cached,
        cache_creation_input_tokens=0,
    )


@dataclass
class GeminiSession:
    """Multi-turn Gemini conversation via `previous_interaction_id`."""
    _sdk_client: Any
    _model: str
    _thinking_level: str
    _max_tokens: int
    _temperature: float
    _system: str
    _prev_interaction_id: str | None = None

    def send(self, user_message: str, *, cache: bool = False) -> LLMResponse:
        kwargs: dict = dict(
            model=self._model,
            input=user_message,
            generation_config={
                "thinking_level": self._thinking_level,
                "thinking_summaries": "auto",
                "max_output_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
        )
        if self._system:
            kwargs["system_instruction"] = self._system
        if self._prev_interaction_id is not None:
            kwargs["previous_interaction_id"] = self._prev_interaction_id

        raw = call_with_retries(self._sdk_client.interactions.create, kwargs=kwargs)
        self._prev_interaction_id = getattr(raw, "id", self._prev_interaction_id)
        return _parse_gemini_response(raw)


@dataclass
class GeminiClient:
    """Wraps `google.genai.Client`. `thinking_level` is one of
    `minimal | low | medium | high` per the Interactions API.
    """
    model: str
    thinking_level: str = "high"
    max_tokens: int = 16000
    temperature: float = 1.0
    _sdk_client: Any = None

    def __post_init__(self):
        if self._sdk_client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._sdk_client = genai.Client(api_key=api_key)

    def start_session(self, *, system: str) -> GeminiSession:
        return GeminiSession(
            _sdk_client=self._sdk_client,
            _model=self.model,
            _thinking_level=self.thinking_level,
            _max_tokens=self.max_tokens,
            _temperature=self.temperature,
            _system=system,
        )
