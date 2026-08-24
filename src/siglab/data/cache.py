"""Local data caching with parquet files."""

from pathlib import Path

import pandas as pd

from siglab import config

_CACHE_DIR: Path | None = None


def _get_cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        cfg = config.load_config()
        _CACHE_DIR = Path(cfg.get("data", {}).get("cache_dir", "data"))
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def set_cache_dir(path: str | Path) -> None:
    """Override the default cache directory."""
    global _CACHE_DIR
    _CACHE_DIR = Path(path)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _path_for(name: str) -> Path:
    return _get_cache_dir() / f"{name}.parquet"


def panel_exists(name: str) -> bool:
    """Check if a cached panel exists."""
    return _path_for(name).exists()


def save_panel(name: str, df: pd.DataFrame) -> None:
    """Save a DataFrame to parquet cache."""
    path = _path_for(name)
    df.to_parquet(path, engine="pyarrow")


def load_panel(name: str) -> pd.DataFrame:
    """Load a DataFrame from parquet cache."""
    path = _path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"Cached panel '{name}' not found at {path}")
    return pd.read_parquet(path, engine="pyarrow")


def save_or_load(name: str, builder_fn, **kwargs) -> pd.DataFrame:
    """Load from cache if available, otherwise build and save.

    Parameters
    ----------
    name : cache key
    builder_fn : callable that returns a DataFrame (called if cache miss)
    **kwargs : passed to builder_fn

    Returns
    -------
    The cached or freshly built DataFrame.
    """
    if panel_exists(name):
        return load_panel(name)
    df = builder_fn(**kwargs)
    save_panel(name, df)
    return df
