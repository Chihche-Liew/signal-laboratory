from __future__ import annotations

import time
from typing import Any, Callable


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}
DEFAULT_MAX_ATTEMPTS = 5


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status in RETRYABLE_STATUS_CODES:
        return True
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "ratelimit", "rate_limit", "connection"))


def call_with_retries(
    call: Callable[..., Any],
    *,
    kwargs: dict[str, Any],
    attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_delay: float = 1.0,
) -> Any:
    delay = initial_delay
    for attempt in range(1, attempts + 1):
        try:
            return call(**kwargs)
        except Exception as exc:
            if attempt >= attempts or not _is_retryable(exc):
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable retry state")
