"""Process-backed evaluation for pandas-heavy signal batches.

The parent process builds the large, read-only ``SignalEvaluator`` once.  On
platforms that support ``fork``, workers inherit those panels copy-on-write;
the pool is deliberately started before pair-level LLM threads are created.
Each proposal is evaluated independently, while results are collected in the
original proposal order so archive and recorder semantics do not change.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

from siglab.lab.archive import EvaluatedSignal, Proposal
from siglab.lab.evaluator.signal_agent_adapter import SignalAgentAdapter


_EVALUATION_AGENT: Any | None = None
_THREADPOOL_LIMITER: Any | None = None
_EVALUATION_EVENTS: Any | None = None


@dataclass(frozen=True)
class EvaluationRequest:
    """Pickle-friendly description of one proposal evaluation."""

    pair: str
    proposal: Proposal
    generation: int
    exclude_financials: bool
    exclude_microcap: bool
    alpha_factor_model: int


@dataclass(frozen=True)
class EvaluationActivity:
    """The proposal currently executing in one evaluator worker."""

    pair: str
    generation: int
    proposal_name: str


class _ActivityMonitor:
    """Track worker start/finish events without involving experiment artifacts."""

    def __init__(self, event_queue: Any) -> None:
        self._event_queue = event_queue
        self._condition = threading.Condition()
        self._active: dict[int, EvaluationActivity] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="siglab-evaluation-activity",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            event = self._event_queue.get()
            kind = event[0]
            if kind == "stop":
                return
            pid = int(event[1])
            with self._condition:
                if kind == "start":
                    self._active[pid] = EvaluationActivity(
                        pair=event[2],
                        generation=int(event[3]),
                        proposal_name=event[4],
                    )
                elif kind == "finish":
                    self._active.pop(pid, None)
                self._condition.notify_all()

    def snapshot(
        self,
        pids: set[int],
        *,
        timeout_seconds: float = 0.25,
    ) -> dict[int, EvaluationActivity]:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while not pids.issubset(self._active) and time.monotonic() < deadline:
                self._condition.wait(deadline - time.monotonic())
            return {
                pid: self._active[pid]
                for pid in pids
                if pid in self._active
            }

    def stop(self) -> None:
        if self._thread is None:
            return
        self._event_queue.put(("stop",))
        self._thread.join(timeout=2.0)
        self._thread = None


def _initialize_worker(spawn_agent: Any | None, event_queue: Any) -> None:
    """Install the evaluator in a worker and prevent nested BLAS fan-out."""
    global _EVALUATION_AGENT, _EVALUATION_EVENTS, _THREADPOOL_LIMITER
    if spawn_agent is not None:
        _EVALUATION_AGENT = spawn_agent
    _EVALUATION_EVENTS = event_queue
    if _EVALUATION_AGENT is None:
        raise RuntimeError("evaluation worker started without a SignalEvaluator")

    # One CPU-bound process should own one BLAS thread.  Without this limit,
    # four evaluator processes can each create a native thread pool and erase
    # the intended process-level speedup through oversubscription.
    from threadpoolctl import threadpool_limits

    _THREADPOOL_LIMITER = threadpool_limits(limits=1)


def _worker_identity() -> int:
    if _EVALUATION_AGENT is None:
        raise RuntimeError("evaluation worker is not initialized")
    return os.getpid()


def _evaluate_request(request: EvaluationRequest) -> EvaluatedSignal:
    if _EVALUATION_AGENT is None:
        raise RuntimeError("evaluation worker is not initialized")
    pid = os.getpid()
    if _EVALUATION_EVENTS is not None:
        _EVALUATION_EVENTS.put(
            (
                "start",
                pid,
                request.pair,
                request.generation,
                request.proposal.name,
            )
        )
    try:
        adapter = SignalAgentAdapter(
            agent=_EVALUATION_AGENT,
            exclude_financials=request.exclude_financials,
            exclude_microcap=request.exclude_microcap,
            alpha_factor_model=request.alpha_factor_model,
        )
        return adapter.evaluate(
            [request.proposal],
            generation=request.generation,
        )[0]
    finally:
        if _EVALUATION_EVENTS is not None:
            _EVALUATION_EVENTS.put(("finish", pid))


class ProcessPoolEvaluator:
    """Evaluate proposals in processes and restore their input ordering."""

    def __init__(self, inner: Any, executor: Executor, *, pair: str):
        self._executor = executor
        self._pair = pair
        self._exclude_financials = bool(inner.exclude_financials)
        self._exclude_microcap = bool(inner.exclude_microcap)
        self._alpha_factor_model = int(inner.alpha_factor_model)

    def evaluate(
        self,
        proposals: Sequence[Proposal],
        generation: int,
    ) -> list[EvaluatedSignal]:
        requests = [
            EvaluationRequest(
                pair=self._pair,
                proposal=proposal,
                generation=generation,
                exclude_financials=self._exclude_financials,
                exclude_microcap=self._exclude_microcap,
                alpha_factor_model=self._alpha_factor_model,
            )
            for proposal in proposals
        ]
        futures = [
            self._executor.submit(_evaluate_request, request)
            for request in requests
        ]
        try:
            # Future completion order is intentionally ignored.  Recorder and
            # archive ordering remain identical to SignalAgentAdapter.evaluate.
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


class EvaluationProcessPool:
    """Own a process executor initialized from one read-only SignalEvaluator."""

    def __init__(
        self,
        *,
        agent: Any,
        max_workers: int,
        start_method: str | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("eval_workers must be at least 1")
        self.agent = agent
        self.max_workers = max_workers
        self.start_method = start_method or self._preferred_start_method()
        self._executor: ProcessPoolExecutor | None = None
        self._event_queue: Any | None = None
        self._activity_monitor: _ActivityMonitor | None = None
        self.worker_pids: tuple[int, ...] = ()

    @staticmethod
    def _preferred_start_method() -> str:
        methods = mp.get_all_start_methods()
        return "fork" if "fork" in methods else mp.get_start_method()

    def start(self) -> "EvaluationProcessPool":
        if self._executor is not None:
            return self

        global _EVALUATION_AGENT, _EVALUATION_EVENTS
        context = mp.get_context(self.start_method)
        self._event_queue = context.SimpleQueue()
        if self.start_method == "fork":
            # Workers inherit the large pandas panels copy-on-write.  run_lab
            # calls start() before creating any pair-level threads.
            _EVALUATION_AGENT = self.agent
            _EVALUATION_EVENTS = self._event_queue
            spawn_agent = None
        else:
            # Portable fallback.  This pickles the evaluator once per worker
            # and therefore has a larger memory footprint than the fork path.
            spawn_agent = self.agent

        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(spawn_agent, self._event_queue),
        )
        warmups = [
            self._executor.submit(_worker_identity)
            for _ in range(self.max_workers)
        ]
        try:
            worker_pids = {future.result() for future in warmups}
            # ProcessPoolExecutor launches every fork worker before starting
            # its management thread. Fast warm-up tasks may nevertheless all
            # execute in one worker, so include the complete live process set.
            worker_pids.update(self._executor._processes)
            self.worker_pids = tuple(sorted(worker_pids))
            self._activity_monitor = _ActivityMonitor(self._event_queue)
            self._activity_monitor.start()
        except BaseException:
            self.close()
            raise
        return self

    def wrap(self, evaluator: Any, *, pair: str) -> ProcessPoolEvaluator:
        if self._executor is None:
            raise RuntimeError("evaluation process pool has not been started")
        return ProcessPoolEvaluator(evaluator, self._executor, pair=pair)

    @staticmethod
    def _exit_description(exitcode: int | None) -> str:
        if exitcode is None:
            return "unknown"
        if exitcode >= 0:
            return str(exitcode)
        try:
            return f"{exitcode} ({signal.Signals(-exitcode).name})"
        except ValueError:
            return str(exitcode)

    def broken_worker_failures(self) -> list[dict[str, Any]]:
        """Describe dead evaluator workers and their last active proposals."""
        if self._executor is None:
            return []
        processes = dict(self._executor._processes or {})
        dead = {
            pid: process.exitcode
            for pid, process in processes.items()
            if process.exitcode is not None
        }
        if not dead:
            return []
        activities = (
            self._activity_monitor.snapshot(set(dead))
            if self._activity_monitor is not None
            else {}
        )
        non_terminated = {
            pid for pid, exitcode in dead.items() if exitcode != -signal.SIGTERM
        }
        suspected = non_terminated or set(dead)
        return [
            {
                "worker_pid": pid,
                "exitcode": exitcode,
                "exit_description": self._exit_description(exitcode),
                "pair": activities[pid].pair if pid in activities else None,
                "generation": (
                    activities[pid].generation if pid in activities else None
                ),
                "proposal_name": (
                    activities[pid].proposal_name if pid in activities else None
                ),
                "suspected_culprit": pid in suspected,
            }
            for pid, exitcode in sorted(dead.items())
        ]

    def describe_broken_pool(
        self,
        failures: list[dict[str, Any]] | None = None,
    ) -> str:
        failures = self.broken_worker_failures() if failures is None else failures
        if not failures:
            return "evaluator worker exited; active pair could not be identified"
        suspected = [row for row in failures if row["suspected_culprit"]]
        rows = suspected or failures
        descriptions = []
        for row in rows:
            context = (
                f"pair={row['pair']} generation={row['generation']} "
                f"proposal={row['proposal_name']!r}"
                if row["pair"] is not None
                else "active pair unknown"
            )
            descriptions.append(
                f"worker pid={row['worker_pid']} exited "
                f"{row['exit_description']} while {context}"
            )
        return "; ".join(descriptions)

    def close(self) -> None:
        global _EVALUATION_AGENT, _EVALUATION_EVENTS
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        if self._activity_monitor is not None:
            self._activity_monitor.stop()
            self._activity_monitor = None
        if self._event_queue is not None:
            self._event_queue.close()
            self._event_queue = None
        if _EVALUATION_AGENT is self.agent:
            _EVALUATION_AGENT = None
        _EVALUATION_EVENTS = None

    def __enter__(self) -> "EvaluationProcessPool":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


__all__ = [
    "EvaluationProcessPool",
    "EvaluationActivity",
    "EvaluationRequest",
    "ProcessPoolEvaluator",
]
