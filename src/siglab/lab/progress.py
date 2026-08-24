from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ProgressRecorder:
    output_dir: Path
    pair_name: str
    run_name: str | None = None
    _output_locks: ClassVar[dict[Path, threading.Lock]] = {}
    _output_locks_guard: ClassVar[threading.Lock] = threading.Lock()
    _started_at: float = field(default_factory=time.monotonic, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _output_key: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._output_key = self.output_dir.resolve()

    @property
    def progress_path(self) -> Path:
        return self.output_dir / "progress.jsonl"

    @property
    def status_path(self) -> Path:
        return self.output_dir / "status.json"

    def record(
        self,
        event: str,
        *,
        stage: str,
        generation: int | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "1",
            "created_at": _utc_now(),
            "elapsed_sec": round(time.monotonic() - self._started_at, 3),
            "run_name": self.run_name,
            "pair": self.pair_name,
            "generation": generation,
            "stage": stage,
            "event": event,
            "message": message,
            "metadata": metadata or {},
        }
        with self._output_lock():
            with self._lock:
                with self.progress_path.open("a") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
                self._write_status(payload)
        return payload

    def _output_lock(self) -> threading.Lock:
        with self._output_locks_guard:
            lock = self._output_locks.get(self._output_key)
            if lock is None:
                lock = threading.Lock()
                self._output_locks[self._output_key] = lock
            return lock

    def _write_status(self, payload: dict[str, Any]) -> None:
        tmp_path = self.output_dir / (
            f"status.{id(self)}.{threading.get_ident()}.{time.monotonic_ns()}.json.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(self.status_path)
