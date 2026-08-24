"""Strict JSON helpers for research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def sanitize_json(value):
    """Recursively convert NumPy scalars and non-finite floats for JSON."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [sanitize_json(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    return value


def strict_json_dumps(value, **kwargs) -> str:
    """Serialize standards-compliant JSON; bare NaN/Infinity are forbidden."""
    return json.dumps(sanitize_json(value), allow_nan=False, **kwargs)


def write_strict_json(path: Path, value, **kwargs) -> None:
    """Write a standards-compliant JSON artifact."""
    path.write_text(strict_json_dumps(value, **kwargs))
