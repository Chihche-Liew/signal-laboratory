"""Factory that builds an LLMClient from a YAML-dict config."""
from __future__ import annotations

from siglab.lab.llm.base import LLMClient
from siglab.lab.llm.anthropic import AnthropicClient
from siglab.lab.llm.mock import MockLLMClient
from siglab.lab.llm.openai import OpenAIClient
from siglab.lab.llm.gemini import GeminiClient


ANTHROPIC_REASONING_EFFORTS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

OPENAI_REASONING_EFFORTS = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

GEMINI_THINKING_LEVELS = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "gemini": "gemini-3-pro",
    "mock": "mock",
}


def _reasoning_effort(cfg: dict) -> str | None:
    effort = cfg.get("reasoning_effort")
    if effort is None:
        return None
    effort = str(effort)
    if effort not in OPENAI_REASONING_EFFORTS:
        known = ", ".join(OPENAI_REASONING_EFFORTS)
        raise ValueError(f"Unknown reasoning_effort {effort!r}; expected one of {known}")
    return effort


def default_model_for_provider(provider: str | None) -> str:
    if provider in DEFAULT_MODELS:
        return DEFAULT_MODELS[provider]
    raise ValueError(f"Unknown LLM provider or missing default model: {provider!r}")


def _model(cfg: dict) -> str:
    if cfg.get("model"):
        return str(cfg["model"])
    return default_model_for_provider(cfg.get("provider"))


def _max_tokens(cfg: dict) -> int:
    if cfg.get("max_tokens") is None:
        return 16000
    return int(cfg["max_tokens"])


def build_llm(cfg: dict, **overrides) -> LLMClient:
    """Return an LLMClient from a dict.

    Supported providers:
      - "anthropic": AnthropicClient (Messages API + extended thinking + prompt cache)
      - "openai":    OpenAIClient (Responses API + reasoning effort + auto prefix cache)
      - "gemini":    GeminiClient (Interactions API + thinking level)
      - "mock":      MockLLMClient (tests)

    `overrides` pass through to the concrete class constructor so tests can
    inject `_sdk_client=MagicMock()`.
    """
    provider = cfg.get("provider")
    effort = _reasoning_effort(cfg)
    model = _model(cfg)
    max_tokens = _max_tokens(cfg)

    if provider == "anthropic":
        thinking = cfg.get("thinking", {})
        output_config = cfg.get("output_config", {})
        return AnthropicClient(
            model=model,
            max_tokens=max_tokens,
            thinking_budget=thinking.get("budget_tokens", 10000),
            thinking_type=thinking.get(
                "type",
                "adaptive",
            ),
            output_effort=output_config.get(
                "effort",
                ANTHROPIC_REASONING_EFFORTS.get(effort),
            ),
            temperature=cfg.get("temperature", 1.0),
            **overrides,
        )

    if provider == "openai":
        reasoning = cfg.get("reasoning", {})
        return OpenAIClient(
            model=model,
            reasoning_effort=reasoning.get(
                "effort",
                OPENAI_REASONING_EFFORTS.get(effort, "xhigh"),
            ),
            max_tokens=max_tokens,
            temperature=cfg.get("temperature", 1.0),
            **overrides,
        )

    if provider == "gemini":
        thinking = cfg.get("thinking", {})
        return GeminiClient(
            model=model,
            thinking_level=thinking.get(
                "level",
                GEMINI_THINKING_LEVELS.get(effort, "high"),
            ),
            max_tokens=max_tokens,
            temperature=cfg.get("temperature", 1.0),
            **overrides,
        )

    if provider == "mock":
        return MockLLMClient(responses=list(cfg.get("responses", [])))

    raise ValueError(f"Unknown LLM provider: {provider!r}")
