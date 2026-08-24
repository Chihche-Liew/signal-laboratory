"""OpenAI SDK wrapper: OpenAIClient + OpenAISession (Responses API).

Multi-turn state is held server-side via `previous_response_id`; the session
stores only the last response id locally. System prompt is passed via
`instructions=` on every turn because OpenAI does not carry previous
instructions forward across `previous_response_id` calls.

Reasoning is enabled on every call via `reasoning={"effort": <level>}`. For
non-`none` efforts, `summary="auto"` opts into text reasoning summaries, which
go into
`LLMResponse.thinking`.

Caching: the Responses API uses automatic prefix caching. The `cache` kwarg
on `session.send()` is a no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from siglab.lab.llm.base import LLMResponse
from siglab.lab.llm.retry import call_with_retries


def _extract_text_and_reasoning(raw) -> tuple[str, str]:
    text = getattr(raw, "output_text", "") or ""
    reasoning_chunks: list[str] = []
    for item in getattr(raw, "output", []) or []:
        itype = getattr(item, "type", None)
        if itype == "reasoning":
            for s in getattr(item, "summary", []) or []:
                txt = getattr(s, "text", None)
                if txt:
                    reasoning_chunks.append(txt)
        elif itype == "message" and not text:
            for c in getattr(item, "content", []) or []:
                txt = getattr(c, "text", None)
                if txt:
                    text += txt
    return text, "\n\n".join(reasoning_chunks)


def _extract_usage(raw) -> tuple[int, int, int]:
    """Return (input_tokens_excl_cached, output_tokens, cached_tokens).

    OpenAI reports `usage.input_tokens` as total-including-cached; we
    subtract to match the Anthropic semantic where `input_tokens` excludes
    cached-read tokens (which live in `cached_input_tokens`).
    """
    u = getattr(raw, "usage", None)
    if u is None:
        return 0, 0, 0
    total_in = getattr(u, "input_tokens", 0) or 0
    out_tok = getattr(u, "output_tokens", 0) or 0
    details = getattr(u, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    cached = cached or 0
    return max(0, total_in - cached), out_tok, cached


def _parse_openai_response(raw) -> LLMResponse:
    text, thinking = _extract_text_and_reasoning(raw)
    in_tok, out_tok, cached = _extract_usage(raw)
    return LLMResponse(
        text=text,
        thinking=thinking,
        raw_content=[],
        input_tokens=in_tok,
        output_tokens=out_tok,
        cached_input_tokens=cached,
        cache_creation_input_tokens=0,
    )


def _reasoning_config(effort: str) -> dict[str, str]:
    cfg = {"effort": effort}
    if effort != "none":
        cfg["summary"] = "auto"
    return cfg


@dataclass
class OpenAISession:
    """Multi-turn OpenAI conversation via `previous_response_id`."""
    _sdk_client: Any
    _model: str
    _reasoning_effort: str
    _max_tokens: int
    _temperature: float
    _system: str
    _prev_response_id: str | None = None

    def send(self, user_message: str, *, cache: bool = False) -> LLMResponse:
        kwargs: dict = dict(
            model=self._model,
            input=user_message,
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
            reasoning=_reasoning_config(self._reasoning_effort),
        )
        if self._system:
            kwargs["instructions"] = self._system
        if self._prev_response_id is not None:
            kwargs["previous_response_id"] = self._prev_response_id

        raw = call_with_retries(self._sdk_client.responses.create, kwargs=kwargs)
        self._prev_response_id = getattr(raw, "id", self._prev_response_id)
        return _parse_openai_response(raw)


@dataclass
class OpenAIClient:
    """Wraps `openai.OpenAI` against the Responses API."""
    model: str
    reasoning_effort: str = "xhigh"
    max_tokens: int = 16000
    temperature: float = 1.0
    _sdk_client: Any = None

    def __post_init__(self):
        if self._sdk_client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._sdk_client = OpenAI(api_key=api_key)

    def start_session(self, *, system: str) -> OpenAISession:
        return OpenAISession(
            _sdk_client=self._sdk_client,
            _model=self.model,
            _reasoning_effort=self.reasoning_effort,
            _max_tokens=self.max_tokens,
            _temperature=self.temperature,
            _system=system,
        )
