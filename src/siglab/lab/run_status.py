from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ParallelRunStatus:
    """Aggregate monitor for all pair jobs in one parallel run."""

    def __init__(self, output_root: Path, run_name: str, timestamp: str) -> None:
        self.output_root = Path(output_root)
        self.run_name = run_name
        self.timestamp = timestamp
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_root / f"{timestamp}_{run_name}_run_status.json"
        self._lock = threading.Lock()
        self._pairs: dict[str, dict[str, Any]] = {}

    def register_pair(self, pair: str, experiment_dir: Path) -> None:
        now = _utc_now()
        with self._lock:
            self._pairs[pair] = {
                "pair": pair,
                "experiment_dir": str(Path(experiment_dir)),
                "state": "queued",
                "stage": "queued",
                "generation": None,
                "updated_at": now,
                "message": None,
                "metadata": {},
            }
            self._write_locked(now)

    def mark_pair(
        self,
        pair: str,
        state: str,
        stage: str,
        generation: int | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        with self._lock:
            row = self._pairs[pair]
            row.update(
                {
                    "state": state,
                    "stage": stage,
                    "generation": generation,
                    "updated_at": now,
                    "message": message,
                    "metadata": metadata or {},
                }
            )
            self._write_locked(now)

    def _write_locked(self, updated_at: str) -> None:
        payload = {
            "schema_version": "1",
            "updated_at": updated_at,
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "totals": self._totals_locked(),
            "pairs": dict(sorted(self._pairs.items())),
        }
        tmp_path = self.output_root / (
            f"{self.status_path.name}.{id(self)}.{threading.get_ident()}."
            f"{time.monotonic_ns()}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(self.status_path)

    def _totals_locked(self) -> dict[str, int]:
        totals = {
            "registered": len(self._pairs),
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        for row in self._pairs.values():
            state = row["state"]
            if state in totals:
                totals[state] += 1
        return totals
