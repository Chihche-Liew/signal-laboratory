from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from siglab.agent.operators import OPERATOR_CATALOG
from siglab.agent.themes import THEMES
from siglab.agent.variables import VARIABLE_CATALOG
from siglab.lab.archive import Proposal, SignalArchive
from siglab.lab.task import DiscoveryTask


def normalize_expression(expression: str) -> str:
    return "".join(expression.lower().split())


# ── Parameter rules ──────────────────────────────────────────────────────
#
# Optional-parameter constraints per operator, applied after the arity check
# in _validate_expr. Each rule receives the operator name, the parameter
# name, and the raw (unparsed) argument string, and returns (ok, message).
# Operators without an entry keep the legacy behavior: any parameter that
# parses as an int is accepted.

ParamRule = Callable[[str, str, str], tuple[bool, str]]


def _min_int(min_value: int) -> ParamRule:
    def _check(op_name: str, param_name: str, raw: str) -> tuple[bool, str]:
        raw = raw.strip()
        try:
            value = int(raw)
        except ValueError:
            return False, f"{op_name} parameter '{param_name}' must be an integer: {raw}"
        if value < min_value:
            return (
                False,
                f"{op_name} parameter '{param_name}' must be >= {min_value}: {raw}",
            )
        return True, ""

    return _check


def _winsor_p(op_name: str, param_name: str, raw: str) -> tuple[bool, str]:
    raw = raw.strip()
    try:
        value = float(raw)
    except ValueError:
        return False, f"{op_name} parameter '{param_name}' must be numeric: {raw}"
    if not (0 < value < 50):
        return False, f"{op_name} parameter '{param_name}' must be in (0, 50): {raw}"
    return True, ""


def _numeric_threshold(op_name: str, param_name: str, raw: str) -> tuple[bool, str]:
    raw = raw.strip()
    try:
        float(raw)
    except ValueError:
        return False, f"{op_name} parameter '{param_name}' must be numeric: {raw}"
    return True, ""


PARAM_RULES: dict[str, ParamRule] = {
    "GROWTH": _min_int(1),
    "DELTA": _min_int(1),
    "LAG": _min_int(1),
    "MA": _min_int(1),
    "VOL": _min_int(2),
    "TREND": _min_int(3),
    "WINSOR": _winsor_p,
    "INDICATOR": _numeric_threshold,
    "TS_MIN": _min_int(1),
    "TS_MAX": _min_int(1),
    "TS_SUM": _min_int(1),
    "TS_COUNT": _min_int(1),
    "TS_RANK": _min_int(2),
}


@dataclass(frozen=True)
class ValidationDiagnostic:
    severity: str
    code: str
    message: str
    proposal_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    accepted: list[Proposal]
    diagnostics: list[ValidationDiagnostic]


@dataclass(frozen=True)
class ExpressionValidation:
    ok: bool
    message: str = ""
    variables: frozenset[str] = frozenset()


def _split_args(args_str: str) -> list[str]:
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for char in args_str:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            if depth < 0:
                return []
            current.append(char)
        elif char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        args.append("".join(current).strip())
    return args


def _balanced_parentheses(expression: str) -> bool:
    depth = 0
    for char in expression:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def validate_expression_syntax(expression: str) -> ExpressionValidation:
    expr = expression.strip()
    if not expr:
        return ExpressionValidation(False, "empty expression")
    if not _balanced_parentheses(expr):
        return ExpressionValidation(False, "unbalanced parentheses")
    return _validate_expr(expr)


def _validate_expr(expr: str) -> ExpressionValidation:
    expr = expr.strip()
    if not expr:
        return ExpressionValidation(False, "empty expression")

    try:
        float(expr)
        return ExpressionValidation(True)
    except ValueError:
        pass

    if re.fullmatch(r"[A-Za-z_]\w*", expr):
        name = expr.lower()
        if name not in VARIABLE_CATALOG:
            return ExpressionValidation(False, f"unknown variable: {expr}")
        return ExpressionValidation(True, variables=frozenset({name}))

    match = re.fullmatch(r"([A-Za-z_]\w*)\((.*)\)", expr, flags=re.DOTALL)
    if not match:
        return ExpressionValidation(False, f"could not parse expression: {expr}")

    op_name = match.group(1).upper()
    op = OPERATOR_CATALOG.get(op_name)
    if op is None:
        return ExpressionValidation(False, f"unknown operator: {op_name}")

    args = _split_args(match.group(2))
    if not args:
        return ExpressionValidation(False, f"{op_name} has no arguments")
    min_args = op.arity
    max_args = op.arity + len(op.params)
    if len(args) < min_args or len(args) > max_args:
        return ExpressionValidation(
            False,
            f"{op_name} expects {min_args}"
            + (f"-{max_args}" if max_args != min_args else "")
            + f" arguments, got {len(args)}",
        )

    variables: set[str] = set()
    for arg in args[: op.arity]:
        result = _validate_expr(arg)
        if not result.ok:
            return result
        variables.update(result.variables)

    rule = PARAM_RULES.get(op_name)
    for arg, param_name in zip(args[op.arity:], op.params):
        if rule is not None:
            ok, message = rule(op_name, param_name, arg)
            if not ok:
                return ExpressionValidation(False, message)
        else:
            try:
                int(arg.strip())
            except ValueError:
                return ExpressionValidation(
                    False,
                    f"{op_name} parameter must be numeric: {arg}",
                )

    return ExpressionValidation(True, variables=frozenset(variables))


def _required_themes(task: DiscoveryTask) -> list[str]:
    return [
        constraint.value
        for constraint in task.constraints
        if constraint.kind == "must_use_theme"
    ]


def _variables_for_theme(theme: str) -> set[str]:
    if theme not in THEMES:
        return set()
    return {variable.lower() for variable in THEMES[theme].variables}


@dataclass
class ProposalValidator:
    reject_invalid_expression: bool = True
    reject_single_theme: bool = True

    def validate(
        self,
        proposals: list[Proposal],
        *,
        task: DiscoveryTask,
        archive: SignalArchive,
    ) -> ValidationResult:
        seen_normalized = {
            normalize_expression(signal.expression)
            for signal in [*archive.considered, *archive.evaluated]
        }
        seen_exact = {
            signal.expression
            for signal in [*archive.considered, *archive.evaluated]
        }
        accepted: list[Proposal] = []
        diagnostics: list[ValidationDiagnostic] = []

        for proposal in proposals:
            normalized_expression = normalize_expression(proposal.expression)
            metadata = {"normalized_expression": normalized_expression}

            missing_field = None
            for field_name in task.output_schema.required_fields:
                value = getattr(proposal, field_name, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_field = field_name
                    break
            if missing_field is not None:
                diagnostics.append(
                    ValidationDiagnostic(
                        severity="error",
                        code="missing_required_field",
                        message=f"required field is empty: {missing_field}",
                        proposal_name=proposal.name,
                        metadata={**metadata, "field": missing_field},
                    )
                )
                continue

            if not proposal.expected_sign or (
                proposal.expected_sign not in task.output_schema.allowed_expected_signs
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        severity="error",
                        code="invalid_expected_sign",
                        message=(
                            "expected_sign must be one of "
                            + ", ".join(task.output_schema.allowed_expected_signs)
                        ),
                        proposal_name=proposal.name,
                        metadata=metadata,
                    )
                )
                continue

            if not proposal.interaction_type or (
                proposal.interaction_type
                not in task.output_schema.allowed_interaction_types
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        severity="error",
                        code="invalid_interaction_type",
                        message=(
                            "interaction_type must be one of "
                            + ", ".join(task.output_schema.allowed_interaction_types)
                        ),
                        proposal_name=proposal.name,
                        metadata=metadata,
                    )
                )
                continue

            expression_check = validate_expression_syntax(proposal.expression)
            metadata["variables"] = sorted(expression_check.variables)
            if self.reject_invalid_expression and not expression_check.ok:
                diagnostics.append(
                    ValidationDiagnostic(
                        severity="error",
                        code="invalid_expression",
                        message=expression_check.message,
                        proposal_name=proposal.name,
                        metadata=metadata,
                    )
                )
                continue

            if self.reject_single_theme and expression_check.ok:
                required = _required_themes(task)
                rejected_theme = None
                for theme in required:
                    if len(required) > 1:
                        others: set[str] = set().union(
                            *(
                                _variables_for_theme(other)
                                for other in required
                                if other != theme
                            )
                        )
                    else:
                        others = set()
                    exclusive = _variables_for_theme(theme) - others
                    if not expression_check.variables.intersection(exclusive):
                        rejected_theme = theme
                        break
                if rejected_theme is not None:
                    diagnostics.append(
                        ValidationDiagnostic(
                            severity="error",
                            code="missing_required_theme",
                            message=(
                                "proposal expression does not use any variable "
                                f"exclusive to required theme: {rejected_theme} "
                                "(variables shared across the paired themes, "
                                "e.g. common denominators like 'at', do not count "
                                "as evidence for either theme)"
                            ),
                            proposal_name=proposal.name,
                            metadata={**metadata, "missing_theme": rejected_theme},
                        )
                    )
                    continue

            if (
                task.novelty_policy.reject_exact_expression_duplicates
                and proposal.expression in seen_exact
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        severity="warning",
                        code="duplicate_expression",
                        message="proposal expression duplicates a prior expression exactly",
                        proposal_name=proposal.name,
                        metadata=metadata,
                    )
                )
                continue

            if (
                task.novelty_policy.reject_normalized_expression_duplicates
                and normalized_expression in seen_normalized
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        severity="warning",
                        code="duplicate_expression",
                        message=(
                            "proposal expression duplicates a prior considered "
                            "or evaluated signal"
                        ),
                        proposal_name=proposal.name,
                        metadata=metadata,
                    )
                )
                continue

            accepted.append(proposal)
            diagnostics.append(
                ValidationDiagnostic(
                    severity="info",
                    code="accepted",
                    message="proposal accepted for evaluation",
                    proposal_name=proposal.name,
                    metadata=metadata,
                )
            )
            seen_normalized.add(normalized_expression)
            seen_exact.add(proposal.expression)

        return ValidationResult(accepted=accepted, diagnostics=diagnostics)
