"""DiscoveryLoop — ties Proposer, Evaluator, Stopping, Budget, Recorder."""
from __future__ import annotations

import inspect
from dataclasses import dataclass

from siglab.lab.archive import SignalArchive
from siglab.lab.budget import Budget
from siglab.lab.evaluator.base import Evaluator
from siglab.lab.proposer.base import Proposer
from siglab.lab.recorder import Recorder
from siglab.lab.stopping.base import StoppingRule
from siglab.lab.task import DiscoveryTask
from siglab.lab.validation import ProposalValidator, normalize_expression


@dataclass
class DiscoveryLoop:
    proposer: Proposer
    evaluator: Evaluator
    stopping: StoppingRule
    budget: Budget
    recorder: Recorder
    validator: ProposalValidator | None = None

    def run(
        self, *, task: DiscoveryTask, n_proposals: int = 5,
    ) -> SignalArchive:
        """Drive propose -> validate -> evaluate -> persist per generation.

        Invariant: every candidate a proposer GENERATES in a generation —
        selected or not — reaches archive.add_considered (order-preserving,
        deduped, keyed by normalize_expression — the validator's
        cross-generation key), so cross-generation dedup and search-universe
        counts mean the same thing for every topology.
        """
        archive = SignalArchive(generation=0)
        gen = 0
        record_progress = getattr(self.recorder, "record_progress", None)

        def progress(
            event: str,
            *,
            stage: str,
            generation: int | None = None,
            message: str | None = None,
            metadata: dict[str, object] | None = None,
        ) -> None:
            if record_progress is not None:
                try:
                    record_progress(
                        event,
                        stage=stage,
                        generation=generation,
                        message=message,
                        metadata=metadata,
                    )
                except Exception:
                    pass

        progress(
            "run_started",
            stage="run",
            generation=None,
            metadata={"task_id": task.task_id, "n_proposals": n_proposals},
        )

        try:
            while True:
                decision = self.stopping.should_stop(
                    archive, meta={"generation": gen, "task_id": task.task_id},
                )
                if decision.stop:
                    progress(
                        "run_completed",
                        stage="run",
                        generation=archive.generation,
                        metadata={
                            **_archive_progress_metadata(archive),
                            "stop_reason": decision.reason,
                        },
                    )
                    break
                if self.budget.exceeded():
                    progress(
                        "run_completed",
                        stage="run",
                        generation=archive.generation,
                        message="budget_exceeded",
                        metadata={
                            **_archive_progress_metadata(archive),
                            "budget_reason": self.budget.reason(),
                        },
                    )
                    break

                progress(
                    "generation_started",
                    stage="generation",
                    generation=gen,
                    metadata={"n_proposals": n_proposals},
                )

                intent = _intent_for_task(task, gen)
                progress(
                    "propose_started",
                    stage="propose",
                    generation=gen,
                    metadata={"n_requested": n_proposals},
                )
                propose_kwargs = {
                    "archive": archive,
                    "task": task,
                    "generation": gen,
                    "n": n_proposals,
                    "intent": intent,
                }
                progress_recorder = getattr(self.recorder, "progress", None)
                if (
                    progress_recorder is not None
                    and _proposer_accepts_progress(self.proposer)
                ):
                    propose_kwargs["progress"] = progress_recorder
                result = self.proposer.propose(**propose_kwargs)
                seen_expr: set[str] = set()
                considered_for_trace = []
                for p in [*result.proposals, *result.considered]:
                    key = normalize_expression(p.expression)
                    if key not in seen_expr:
                        seen_expr.add(key)
                        considered_for_trace.append(p)
                progress(
                    "propose_completed",
                    stage="propose",
                    generation=gen,
                    metadata={
                        "n_proposals": len(result.proposals),
                        "n_considered": len(considered_for_trace),
                        "n_parse_diagnostics": len(result.parse_diagnostics),
                        "n_critic_notes": len(result.critic_notes),
                    },
                )

                self.budget.record_call(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cached_input_tokens=result.cached_input_tokens,
                    cache_creation_input_tokens=(
                        result.cache_creation_input_tokens
                    ),
                )

                proposals_for_eval = result.proposals
                validation_diagnostics = []
                progress(
                    "validation_started",
                    stage="validation",
                    generation=gen,
                    metadata={"n_proposals": len(result.proposals)},
                )
                if self.validator is not None:
                    validation = self.validator.validate(
                        result.proposals, task=task, archive=archive,
                    )
                    proposals_for_eval = validation.accepted
                    validation_diagnostics = validation.diagnostics
                progress(
                    "validation_completed",
                    stage="validation",
                    generation=gen,
                    metadata={
                        "n_accepted": len(proposals_for_eval),
                        "n_rejected": (
                            len(result.proposals) - len(proposals_for_eval)
                        ),
                        "n_diagnostics": len(validation_diagnostics),
                    },
                )

                progress(
                    "evaluate_started",
                    stage="evaluate",
                    generation=gen,
                    metadata={"n_proposals": len(proposals_for_eval)},
                )
                eval_results = self.evaluator.evaluate(
                    proposals_for_eval, generation=gen,
                )
                progress(
                    "evaluate_completed",
                    stage="evaluate",
                    generation=gen,
                    metadata={"n_evaluated": len(eval_results)},
                )
                archive.add_evaluated(eval_results)
                archive.add_considered(considered_for_trace)
                archive.add_critic_notes(result.critic_notes)
                if result.moderator_note is not None:
                    archive.add_moderator_note(result.moderator_note)
                archive.add_parse_diagnostics(result.parse_diagnostics)
                archive.add_validation_diagnostics(validation_diagnostics)
                archive.generation = gen

                progress(
                    "generation_persist_started",
                    stage="persist",
                    generation=gen,
                    metadata=_archive_progress_metadata(archive),
                )

                save_validation_diagnostics = getattr(
                    self.recorder, "save_validation_diagnostics", None,
                )
                if save_validation_diagnostics is not None:
                    save_validation_diagnostics(
                        generation=gen, diagnostics=validation_diagnostics,
                    )

                save_parser_diagnostics = getattr(
                    self.recorder, "save_parser_diagnostics", None,
                )
                if save_parser_diagnostics is not None:
                    save_parser_diagnostics(
                        generation=gen, diagnostics=result.parse_diagnostics,
                    )

                self.recorder.save_generation(
                    generation=gen,
                    llm_calls=result.llm_calls,
                    proposals=result.proposals,
                    results=eval_results,
                    considered=considered_for_trace,
                    critic_notes=result.critic_notes,
                    moderator_note=result.moderator_note,
                )
                self.recorder.save_archive(archive)
                progress(
                    "generation_completed",
                    stage="generation",
                    generation=gen,
                    metadata=_archive_progress_metadata(archive),
                )

                gen += 1

            self.recorder.save_archive(archive)
            return archive
        except Exception as exc:
            try:
                progress(
                    "run_failed",
                    stage="failed",
                    generation=gen,
                    message=str(exc),
                    metadata={
                        **_archive_progress_metadata(archive),
                        "error_type": type(exc).__name__,
                    },
                )
            except Exception:
                pass
            raise


def _intent_for_task(task: DiscoveryTask, generation: int) -> str:
    mode = task.metadata.get("mode")
    if mode in {"repair", "explore", "exploit"}:
        return str(mode)
    return "reflect" if generation > 0 else "explore"


def _proposer_accepts_progress(proposer: Proposer) -> bool:
    try:
        parameters = inspect.signature(proposer.propose).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "progress"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _archive_progress_metadata(archive: SignalArchive) -> dict[str, int]:
    return {
        "archive_evaluated": len(archive.evaluated),
        "archive_considered": len(archive.considered),
        "archive_critic_notes": len(archive.critic_notes),
        "archive_parse_diagnostics": len(archive.parse_diagnostics),
        "archive_validation_diagnostics": len(archive.validation_diagnostics),
    }
