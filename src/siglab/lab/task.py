from dataclasses import asdict, dataclass, field
from typing import Any

from siglab.factor_model.models import ALPHA_FACTOR_MODEL_LABELS


@dataclass(frozen=True)
class TaskObjective:
    title: str
    description: str


@dataclass(frozen=True)
class TaskConstraint:
    kind: str
    value: str


@dataclass(frozen=True)
class OutputSchema:
    required_fields: list[str]
    allowed_expected_signs: list[str]
    allowed_interaction_types: list[str]


@dataclass(frozen=True)
class EvaluationProtocol:
    evaluator_type: str
    primary_metric: str
    success_threshold: float
    require_expected_sign_match: bool
    alpha_factor_model: int = 5

    def __post_init__(self) -> None:
        if self.alpha_factor_model not in ALPHA_FACTOR_MODEL_LABELS:
            raise ValueError(
                "alpha_factor_model must be one of "
                f"{sorted(ALPHA_FACTOR_MODEL_LABELS)}"
            )


@dataclass(frozen=True)
class SuccessCriteria:
    min_abs_tstat: float
    require_expected_sign_match: bool
    min_coverage: float


@dataclass(frozen=True)
class NoveltyPolicy:
    reject_exact_expression_duplicates: bool
    reject_normalized_expression_duplicates: bool


@dataclass(frozen=True)
class DiscoveryTask:
    task_id: str
    task_type: str
    objective: TaskObjective
    constraints: list[TaskConstraint]
    output_schema: OutputSchema
    evaluation_protocol: EvaluationProtocol
    success_criteria: SuccessCriteria
    novelty_policy: NoveltyPolicy
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.task_type.strip():
            raise ValueError("task_type must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
