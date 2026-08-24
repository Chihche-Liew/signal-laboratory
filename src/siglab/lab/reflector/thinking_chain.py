"""Reflector that reuses the existing cross_theme prompt builders.

Gen-0 prompts pass through unchanged. Later generations include a generation
budget line so the agent knows how many iterations remain.
"""
from __future__ import annotations

from dataclasses import dataclass

from siglab.lab.archive import SignalArchive
from siglab.lab.context.base import ArchiveSummary
from siglab.lab.prompting import CrossThemePromptRenderer, PromptContext, PromptRole
from siglab.lab.task import DiscoveryTask


def _budget_line(generation: int, max_generations: int) -> str:
    remaining = max_generations - generation - 1
    return (
        f"\n\n=== GENERATION BUDGET ===\n"
        f"You are in generation {generation} of {max_generations}. "
        f"{remaining} generations remain after this one.\n"
        f"Earlier generations should prioritize exploring diverse mechanisms; "
        f"later generations should consolidate and refine the most promising "
        f"directions.\n"
    )


@dataclass
class ThinkingChainReflector:
    """Baseline reflector with persistent thinking-chain context."""
    max_generations: int
    n_proposals: int

    def summarize(
        self, *, archive: SignalArchive, task: DiscoveryTask,
        generation: int,
        intent: str | None = None,
    ) -> ArchiveSummary:
        guidance = _loop_guidance(task)
        active_intent = intent or _intent_for_task(task, generation)
        context = PromptContext(
            task=task,
            generation=generation,
            max_generations=self.max_generations,
            role=PromptRole(name="proposer", instructions=""),
            archive=archive,
            intent=active_intent,
            guidance=guidance,
        )
        rendered = CrossThemePromptRenderer(n_proposals=self.n_proposals).render(context)
        text = rendered.user_message
        if generation > 0:
            text += _budget_line(generation, self.max_generations)
        return ArchiveSummary(
            text=text,
            generation=generation,
            max_generations=self.max_generations,
        )


def _loop_guidance(task: DiscoveryTask) -> str | None:
    seed_prompt = task.metadata.get("seed_prompt")
    guidance_scope = task.metadata.get("guidance_scope")
    if seed_prompt and guidance_scope in {"loop", "task_and_loop"}:
        return str(seed_prompt)
    return None


def _intent_for_task(task: DiscoveryTask, generation: int) -> str:
    mode = task.metadata.get("mode")
    if mode in {"repair", "explore", "exploit"}:
        return str(mode)
    return "reflect" if generation > 0 else "explore"
