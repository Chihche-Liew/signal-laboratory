"""WRDS database connection management.

Reads WRDS_USERNAME from the .env file in the repo root (via python-dotenv),
falling back to ~/.pgpass and finally to an interactive prompt.
"""

import os
from pathlib import Path

import wrds
from dotenv import load_dotenv

# Load .env from repo root (two levels up from this file: src/siglab/data/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")

_conn: wrds.Connection | None = None


def _read_pgpass_parts() -> list[str] | None:
    pgpass = Path(os.path.expanduser("~")) / ".pgpass"
    if not pgpass.exists():
        return None
    try:
        for line in pgpass.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) >= 5:
                return parts
    except Exception:
        return None
    return None


def _resolve_username() -> str | None:
    """Try WRDS_USERNAME env var, then ~/.pgpass."""
    username = os.environ.get("WRDS_USERNAME", "").strip()
    if username:
        return username

    parts = _read_pgpass_parts()
    if parts is not None:
        return parts[3]

    return None


def _resolve_password() -> str | None:
    """Try WRDS_PASSWORD/PGPASSWORD env vars, then ~/.pgpass."""
    for key in ("WRDS_PASSWORD", "PGPASSWORD"):
        password = os.environ.get(key, "").strip()
        if password:
            return password

    parts = _read_pgpass_parts()
    if parts is not None:
        return parts[4]

    return None


def get_connection() -> wrds.Connection:
    """Get or create a cached WRDS connection."""
    global _conn
    if _conn is None:
        username = _resolve_username()
        password = _resolve_password()
        kwargs = {}
        if username:
            kwargs["wrds_username"] = username
        if password:
            kwargs["wrds_password"] = password
        if kwargs:
            _conn = wrds.Connection(**kwargs)
        else:
            _conn = wrds.Connection()  # interactive prompt
    return _conn


def close_connection() -> None:
    """Close the WRDS connection."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def raw_sql(sql: str, **kwargs):
    """Execute raw SQL against WRDS and return a DataFrame."""
    return get_connection().raw_sql(sql, **kwargs)
