from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from siglab.factor_model.models import ALPHA_FACTOR_MODEL_LABELS
from siglab.lab.llm.factory import default_model_for_provider


@dataclass(frozen=True)
class RunIdentity:
    name: str
    output_root: str = "data/experiments"
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("RunIdentity.name must be non-empty")
        if (
            "/" in self.name
            or "\\" in self.name
            or ".." in self.name
            or self.name.startswith(".")
        ):
            raise ValueError(
                f"RunIdentity.name {self.name!r} must be filesystem-safe: "
                "no '/', '\\\\' or '..' and no leading '.' — run.name becomes "
                "a single output-folder component under output_root"
            )


@dataclass(frozen=True)
class ModeSpec:
    type: str = "default"
    seed_prompt: str | None = None
    guidance_scope: str = "none"

    def __post_init__(self) -> None:
        valid_modes = {"default", "guided", "repair", "explore", "exploit"}
        valid_scopes = {"none", "task", "loop", "task_and_loop"}

        if self.type not in valid_modes:
            raise ValueError(f"mode type must be one of {sorted(valid_modes)}")
        if self.guidance_scope not in valid_scopes:
            raise ValueError(f"guidance_scope must be one of {sorted(valid_scopes)}")
        if self.seed_prompt is not None and self.guidance_scope == "none":
            raise ValueError("guidance_scope must not be none when seed_prompt is provided")


@dataclass(frozen=True)
class TaskRunSpec:
    type: str
    pair: str
    theme_a: str | None
    theme_b: str | None
    objective_override: str | None = None


@dataclass(frozen=True)
class LLMSpec:
    provider: str = "anthropic"
    model: str = "claude-opus-4-8"
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    temperature: float = 1.0
    thinking: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposerSpec:
    type: str
    n_proposals: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopSpec:
    n_generations: int
    proposals_per_generation: int
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorSpec:
    sample_filters: tuple[str, ...]
    success_threshold: float
    alpha_factor_model: int = 5

    def __post_init__(self) -> None:
        if self.alpha_factor_model not in ALPHA_FACTOR_MODEL_LABELS:
            raise ValueError(
                "evaluator.alpha_factor_model must be one of "
                f"{sorted(ALPHA_FACTOR_MODEL_LABELS)}"
            )

    @property
    def exclude_financials(self) -> bool:
        return "exclude_financials" in self.sample_filters

    @property
    def exclude_microcap(self) -> bool:
        return "exclude_microcap" in self.sample_filters


@dataclass(frozen=True)
class BudgetSpec:
    max_tokens_in: int | None = None
    max_tokens_out: int | None = None
    max_wall_seconds: int | None = None


@dataclass(frozen=True)
class PosthocSpec:
    enabled: bool
    suites: list[str]


@dataclass(frozen=True)
class RunSpec:
    run: RunIdentity
    mode: ModeSpec
    task: TaskRunSpec
    llm: LLMSpec
    proposer: ProposerSpec
    loop: LoopSpec
    evaluator: EvaluatorSpec
    budget: BudgetSpec
    posthoc: PosthocSpec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def experiment_folder(self, timestamp: str) -> str:
        folder_name = f"{timestamp}_{self.run.name}_{self.task.pair}"
        return str(PurePosixPath(self.run.output_root) / folder_name)


def _mode_from_raw(raw: dict[str, Any] | None) -> ModeSpec:
    raw = raw or {}
    seed_prompt = raw.get("seed_prompt")
    guidance_scope = raw.get("guidance_scope", "none")
    if seed_prompt and guidance_scope == "none":
        guidance_scope = "task_and_loop"
    return ModeSpec(
        type=raw.get("type", "default"),
        seed_prompt=seed_prompt,
        guidance_scope=guidance_scope,
    )


def _task_pairs(raw_task: dict[str, Any], pair_override: str | None) -> list[str]:
    if pair_override:
        return [pair_override]
    if raw_task.get("pair"):
        return [str(raw_task["pair"])]
    return [str(pair) for pair in raw_task.get("pairs", [])]


def _task_spec_from_raw(raw_task: dict[str, Any], pair: str) -> TaskRunSpec:
    theme_a = raw_task.get("theme_a")
    theme_b = raw_task.get("theme_b")
    if raw_task.get("type", "cross_theme") == "cross_theme" and (not theme_a or not theme_b):
        from siglab.lab.tasks.cross_theme import get_cross_theme_task_spec

        pair_spec = get_cross_theme_task_spec(pair)
        theme_a = theme_a or pair_spec.theme_a
        theme_b = theme_b or pair_spec.theme_b
    return TaskRunSpec(
        type=raw_task.get("type", "cross_theme"),
        pair=pair,
        theme_a=theme_a,
        theme_b=theme_b,
        objective_override=raw_task.get("objective_override"),
    )


def _llm_spec_from_raw(raw: dict[str, Any] | None) -> LLMSpec:
    raw = raw or {}
    known = {
        "provider",
        "model",
        "reasoning_effort",
        "max_tokens",
        "temperature",
        "thinking",
    }
    provider = raw.get("provider", "anthropic")
    return LLMSpec(
        provider=provider,
        model=raw.get("model", default_model_for_provider(provider)),
        reasoning_effort=raw.get("reasoning_effort"),
        max_tokens=int(raw["max_tokens"]) if raw.get("max_tokens") is not None else None,
        temperature=float(raw.get("temperature", 1.0)),
        thinking=dict(raw.get("thinking", {})),
        extra={key: value for key, value in raw.items() if key not in known},
    )


def _proposer_spec_from_raw(raw: dict[str, Any]) -> ProposerSpec:
    known = {"type", "n_proposals"}
    return ProposerSpec(
        type=raw.get("type", "single_agent"),
        n_proposals=int(raw.get("n_proposals", 5)),
        extra={key: value for key, value in raw.items() if key not in known},
    )


def _loop_spec_from_raw(
    raw: dict[str, Any],
    proposer: ProposerSpec,
    gens_override: int | None,
) -> LoopSpec:
    n_generations = int(
        gens_override if gens_override is not None else raw.get("n_generations", 7)
    )
    return LoopSpec(
        n_generations=n_generations,
        proposals_per_generation=int(
            raw.get("proposals_per_generation", proposer.n_proposals)
        ),
        validation=dict(raw.get("validation", {})),
    )


def _evaluator_spec_from_raw(raw: dict[str, Any]) -> EvaluatorSpec:
    return EvaluatorSpec(
        sample_filters=_sample_filters_from_raw(raw),
        success_threshold=float(raw.get("success_threshold", 1.96)),
        alpha_factor_model=int(raw.get("alpha_factor_model", 5)),
    )


def _sample_filters_from_raw(raw: dict[str, Any]) -> tuple[str, ...]:
    allowed = {"exclude_financials", "exclude_microcap"}
    if "sample_filters" in raw:
        raw_filters = raw.get("sample_filters") or []
    elif "subsample" in raw:
        raw_filters = raw.get("subsample") or []
    else:
        raw_filters = []
        if bool(raw.get("exclude_financials", True)):
            raw_filters.append("exclude_financials")
        if bool(raw.get("exclude_microcap", True)):
            raw_filters.append("exclude_microcap")

    if isinstance(raw_filters, str):
        raw_filters = [raw_filters]

    filters: list[str] = []
    for item in raw_filters:
        value = str(item)
        if value not in allowed:
            known = ", ".join(sorted(allowed))
            raise ValueError(f"unknown evaluator sample filter {value!r}; expected one of {known}")
        if value not in filters:
            filters.append(value)
    return tuple(filters)


def _budget_spec_from_raw(raw: dict[str, Any] | None) -> BudgetSpec:
    raw = raw or {}
    return BudgetSpec(
        max_tokens_in=(
            int(raw["max_tokens_in"]) if raw.get("max_tokens_in") is not None else None
        ),
        max_tokens_out=(
            int(raw["max_tokens_out"]) if raw.get("max_tokens_out") is not None else None
        ),
        max_wall_seconds=raw.get("max_wall_seconds", raw.get("max_wall_clock_seconds")),
    )


def _posthoc_spec_from_raw(raw: dict[str, Any] | None) -> PosthocSpec:
    raw = raw or {}
    return PosthocSpec(
        enabled=bool(raw.get("enabled", False)),
        suites=list(raw.get("suites", [])),
    )


# RunSpec fields that were parsed and echoed into resolved_config.json but
# never influenced execution (verified per-field at 8fdf920). Accepting them
# silently means a YAML edit no-ops while persisted provenance records it as
# applied — the exact failure mode behind the 2026-07-07 objective-drift
# incident — so a YAML that still carries them fails loudly instead.
_REMOVED_FIELDS: tuple[tuple[str, str], ...] = (
    ("run", "seed"),
    ("task", "reference_library"),
    ("task", "constraints"),
    ("evaluator", "type"),
    ("evaluator", "primary_metric"),
    ("loop", "lesson_retrieval"),
)


def _reject_removed_fields(raw: dict[str, Any]) -> None:
    for section, key in _REMOVED_FIELDS:
        section_raw = raw.get(section) or {}
        if isinstance(section_raw, dict) and key in section_raw:
            raise ValueError(
                f"RunSpec field '{section}.{key}' was never implemented: it "
                "used to be parsed and echoed into resolved_config.json "
                "without ever influencing execution, and has been removed. "
                "Delete it from the YAML."
            )


def load_runspecs(
    path: str | Path,
    *,
    pair_override: str | None = None,
    gens_override: int | None = None,
    output_root_override: str | None = None,
    mode_override: str | None = None,
    seed_prompt: str | None = None,
    guidance_scope: str | None = None,
) -> list[RunSpec]:
    """Load one architecture YAML and expand it to resolved RunSpecs.

    The YAML can specify either `task.pair` or `task.pairs`. The resolved
    contract always contains exactly one pair per RunSpec.
    """
    raw = yaml.safe_load(Path(path).read_text())
    _reject_removed_fields(raw)
    raw_run = raw.get("run", {})
    raw_mode = dict(raw.get("mode", {}))
    raw_task = raw.get("task", {})

    if mode_override is not None:
        raw_mode["type"] = mode_override
    if seed_prompt is not None:
        raw_mode["seed_prompt"] = seed_prompt
    if guidance_scope is not None:
        raw_mode["guidance_scope"] = guidance_scope

    mode = _mode_from_raw(raw_mode)
    llm = _llm_spec_from_raw(raw.get("llm"))
    proposer = _proposer_spec_from_raw(raw["proposer"])
    loop = _loop_spec_from_raw(raw.get("loop", {}), proposer, gens_override)
    evaluator = _evaluator_spec_from_raw(raw.get("evaluator", {}))
    budget = _budget_spec_from_raw(raw.get("budget"))
    posthoc = _posthoc_spec_from_raw(raw.get("posthoc"))

    pairs = _task_pairs(raw_task, pair_override)
    if not pairs:
        raise ValueError("RunSpec YAML must provide task.pair or task.pairs")

    output_root = output_root_override or raw_run.get("output_root", "data/experiments")
    run = RunIdentity(
        name=raw_run["name"],
        output_root=output_root,
        tags=list(raw_run.get("tags", [])),
    )

    return [
        RunSpec(
            run=run,
            mode=mode,
            task=_task_spec_from_raw(raw_task, pair),
            llm=llm,
            proposer=proposer,
            loop=loop,
            evaluator=evaluator,
            budget=budget,
            posthoc=posthoc,
        )
        for pair in pairs
    ]
