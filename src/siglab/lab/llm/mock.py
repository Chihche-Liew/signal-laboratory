"""Deterministic mock LLM client for tests."""
from __future__ import annotations

from dataclasses import dataclass, field

from siglab.lab.llm.base import LLMResponse


@dataclass
class MockLLMSession:
    """Session stub — pops from the parent client's shared response queue
    and records each send on `parent.calls` for test assertions.

    Each entry in `parent.calls` has:
      - system:       the system prompt the session was opened with
      - user_message: the current turn's user input string
      - cache:        the `cache` kwarg (bool) for this send
      - prior_turns:  a list of {role, content} dicts for turns accumulated
                      BEFORE this send (empty on first turn, 2 after first,
                      4 after second, etc.)
    """
    _parent: "MockLLMClient"
    _system: str
    _history: list = field(default_factory=list)

    def send(self, user_message: str, *, cache: bool = False) -> LLMResponse:
        if not self._parent.responses:
            raise RuntimeError("MockLLMClient response queue exhausted")
        text = self._parent.responses.pop(0)
        prior_turns_snapshot = list(self._history)
        self._parent.calls.append({
            "system": self._system,
            "user_message": user_message,
            "cache": cache,
            "prior_turns": prior_turns_snapshot,
        })
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": text})
        return LLMResponse(text=text, raw_content=[{"type": "text", "text": text}])


@dataclass
class MockLLMClient:
    """LLMClient that returns canned responses in order. Tests instantiate
    with a list of response strings; each `session.send()` pops the next
    one. All sends are recorded on `.calls`.
    """
    responses: list[str]
    calls: list[dict] = field(default_factory=list)

    def start_session(self, *, system: str) -> MockLLMSession:
        return MockLLMSession(_parent=self, _system=system)
