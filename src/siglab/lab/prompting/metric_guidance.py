"""Shared prompt guidance for evaluation statistics."""

from __future__ import annotations

from siglab.factor_model.models import ALPHA_FACTOR_MODEL_LABELS
from siglab.lab.task import EvaluationProtocol


def _alpha_model_label(evaluation_protocol: EvaluationProtocol) -> str:
    return ALPHA_FACTOR_MODEL_LABELS[evaluation_protocol.alpha_factor_model]


def format_evaluation_stat_definitions(
    evaluation_protocol: EvaluationProtocol,
) -> str:
    alpha_model = _alpha_model_label(evaluation_protocol)
    return "\n".join([
        "EVALUATION STAT DEFINITIONS",
        (
            "- fmb_tstat: Fama-MacBeth cross-sectional t-stat for the "
            "signal's return-predictive slope; sign should align with "
            "EXPECTED_SIGN."
        ),
        (
            f"- ls_alpha: annualized long-short {alpha_model} alpha for the "
            "signal-sorted portfolio spread."
        ),
        f"- ls_talpha: t-stat of the long-short {alpha_model} alpha.",
        "- ls_sharpe: Sharpe ratio of the long-short portfolio spread.",
        "- coverage: fraction of usable firm-month signal observations.",
    ])


def format_evaluation_search_objective(
    evaluation_protocol: EvaluationProtocol,
) -> str:
    alpha_model = _alpha_model_label(evaluation_protocol)
    return "\n".join([
        "MULTI-METRIC RESEARCH OBJECTIVE",
        (
            "- Cross-sectional evidence: seek an economically coherent "
            "Fama-MacBeth slope with a strong fmb_tstat in the EXPECTED_SIGN "
            "direction."
        ),
        (
            "- Portfolio evidence: seek an economically meaningful annualized "
            f"{alpha_model} long-short alpha with a strong ls_talpha in the "
            "EXPECTED_SIGN direction."
        ),
        (
            "Treat ls_alpha as abnormal-return magnitude and ls_talpha as its "
            "statistical reliability. Do not pursue a large but imprecise alpha, "
            "or a precise but economically negligible alpha."
        ),
        (
            "Prefer mechanisms and refinements that strengthen both forms of "
            "evidence. When they disagree, diagnose the economic or portfolio "
            "reason and use it to motivate a substantively different improvement, "
            "not a cosmetic variant of an earlier signal."
        ),
        (
            "For a negative EXPECTED_SIGN, stronger evidence means more negative "
            "estimates with reliable statistical support; it does not mean a "
            "numerically larger value."
        ),
        (
            "Use coverage and ls_sharpe as supporting diagnostics. Avoid apparent "
            "improvements driven by sparse coverage or unstable portfolio "
            "composition."
        ),
    ])


def format_evaluation_revision_objective(
    evaluation_protocol: EvaluationProtocol,
) -> str:
    alpha_model = _alpha_model_label(evaluation_protocol)
    return "\n".join([
        "REVISION OBJECTIVE",
        (
            "Revise toward joint, EXPECTED_SIGN-consistent strength in fmb_tstat, "
            f"annualized {alpha_model} ls_alpha, and ls_talpha. Preserve a "
            "credible mechanism and adequate coverage rather than improving one "
            "metric by weakening the others."
        ),
    ])
