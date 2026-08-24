from __future__ import annotations

import argparse
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from siglab.lab.archive import SignalArchive
from siglab.lab.budget import Budget
from siglab.lab.evaluator.signal_agent_adapter import SignalAgentAdapter
from siglab.lab.llm.factory import build_llm, default_model_for_provider
from siglab.lab.loop import DiscoveryLoop
from siglab.lab.manifest import RunManifest
from siglab.lab.posthoc import write_posthoc_manifest, write_search_universe_summary
from siglab.lab.proposer.single_agent import SingleAgentProposer
from siglab.lab.recorder import Recorder
from siglab.lab.reflector.thinking_chain import ThinkingChainReflector
from siglab.lab.runspec import ModeSpec, RunSpec
from siglab.lab.stopping.fixed import FixedGens
from siglab.lab.task import DiscoveryTask
from siglab.lab.tasks.cross_theme import build_cross_theme_task
from siglab.lab.validation import ProposalValidator


REPO_ROOT = Path(__file__).resolve().parents[3]
PROPOSER_CRITIC_DEFAULT_PROVIDERS = {
    "proposer": "anthropic",
    "critic": "anthropic",
}


@dataclass
class ExperimentRunResult:
    spec: RunSpec
    task: DiscoveryTask
    archive: SignalArchive
    loop: DiscoveryLoop
    output_dir: Path


def build_mode_spec(args: argparse.Namespace) -> ModeSpec:
    guidance_scope = args.guidance_scope
    if args.seed_prompt and guidance_scope == "none":
        guidance_scope = "task_and_loop"
    return ModeSpec(
        type=args.mode,
        seed_prompt=args.seed_prompt,
        guidance_scope=guidance_scope,
    )


def experiment_dir(spec: RunSpec, timestamp: str) -> Path:
    return Path(spec.experiment_folder(timestamp))


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_task(spec: RunSpec) -> DiscoveryTask:
    if spec.task.type != "cross_theme":
        raise ValueError(f"Unknown task type: {spec.task.type!r}")
    return build_cross_theme_task(
        spec.task.pair,
        mode=spec.mode,
        sample_filters=spec.evaluator.sample_filters,
        objective_override=spec.task.objective_override,
        alpha_factor_model=spec.evaluator.alpha_factor_model,
    )


def build_manifest(
    *,
    spec: RunSpec,
    task: DiscoveryTask,
    experiment_dir: Path,
    timestamp: str,
    git_sha: str | None = None,
) -> RunManifest:
    provider, model, role_llms = _manifest_llm_identity(spec)
    return RunManifest(
        schema_version="1",
        experiment_id=f"{timestamp}_{spec.run.name}_{spec.task.pair}",
        run_name=spec.run.name,
        created_at=timestamp,
        git_sha=git_sha or globals()["git_sha"](),
        provider=provider,
        model=model,
        task_id=task.task_id,
        output_dir=str(experiment_dir),
        mode=asdict(spec.mode),
        reference_library={
            "source_id": task.metadata.get(
                "reference_library",
                "canonical_single_theme_reference",
            ),
            "version": "dev",
        },
        cache_semantics={
            "prompt_cache": "provider_default",
            "thinking": spec.llm.thinking,
            "role_llms": role_llms,
        },
        posthoc=asdict(spec.posthoc),
    )


def resolved_config_payload(
    *,
    spec: RunSpec,
    task: DiscoveryTask,
    experiment_dir: Path,
) -> dict:
    payload = spec.to_dict()
    payload["task_contract"] = task.to_dict()
    payload["output_dir"] = str(experiment_dir)
    return payload


def _llm_config(spec: RunSpec) -> dict[str, Any]:
    return {
        "provider": spec.llm.provider,
        "model": spec.llm.model,
        "reasoning_effort": spec.llm.reasoning_effort,
        "max_tokens": spec.llm.max_tokens,
        "temperature": spec.llm.temperature,
        "thinking": spec.llm.thinking,
        **spec.llm.extra,
    }


def _role_llms_config(spec: RunSpec) -> dict[str, Any]:
    if "llms" not in spec.proposer.extra:
        return {}
    role_llms = spec.proposer.extra["llms"]
    if not isinstance(role_llms, dict):
        raise ValueError("proposer.llms must be a mapping of role names to LLM configs")
    return role_llms


def _role_llm_config(spec: RunSpec, role: str) -> dict[str, Any]:
    role_llms = _role_llms_config(spec)
    role_cfg = role_llms.get(role, {})
    if not isinstance(role_cfg, dict):
        raise ValueError(f"proposer.llms.{role} must be a mapping")

    default_provider = PROPOSER_CRITIC_DEFAULT_PROVIDERS[role]
    provider = role_cfg.get("provider", default_provider)
    cfg = {
        "provider": provider,
        "model": default_model_for_provider(provider),
        "reasoning_effort": spec.llm.reasoning_effort,
        "max_tokens": spec.llm.max_tokens,
        "temperature": spec.llm.temperature,
        "thinking": spec.llm.thinking,
        **spec.llm.extra,
    }
    cfg.update(role_cfg)
    if not cfg.get("model"):
        cfg["model"] = default_model_for_provider(cfg.get("provider"))
    return cfg


def _resolved_role_llms_config(spec: RunSpec) -> dict[str, Any]:
    return {
        "proposer": _role_llm_config(spec, "proposer"),
        "critic": _role_llm_config(spec, "critic"),
    }


def _manifest_llm_identity(spec: RunSpec) -> tuple[str, str, dict[str, Any]]:
    if spec.proposer.type == "proposer_critic":
        return "mixed", "mixed", _resolved_role_llms_config(spec)
    return spec.llm.provider, spec.llm.model, {}


def build_components(
    spec: RunSpec,
    *,
    agent: Any,
    output_dir_override: str | Path | None = None,
    llm_overrides: dict | None = None,
) -> DiscoveryLoop:
    base_n = int(spec.proposer.n_proposals)
    max_gens = int(spec.loop.n_generations)

    ptype = spec.proposer.type
    if ptype == "single_agent":
        llm = build_llm(_llm_config(spec), **(llm_overrides or {}))
        reflector = ThinkingChainReflector(max_generations=max_gens, n_proposals=base_n)
        proposer = SingleAgentProposer(llm=llm, reflector=reflector)
    elif ptype == "proposer_critic":
        from siglab.lab.proposer.proposer_critic import ProposerCriticProposer

        reflector = ThinkingChainReflector(max_generations=max_gens, n_proposals=base_n)
        proposer_llm = build_llm(
            _role_llm_config(spec, "proposer"),
            **(llm_overrides or {}),
        )
        critic_llm = build_llm(
            _role_llm_config(spec, "critic"),
            **(llm_overrides or {}),
        )
        proposer = ProposerCriticProposer(
            proposer_llm=proposer_llm,
            critic_llm=critic_llm,
            reflector=reflector,
        )
    elif ptype == "debate":
        from siglab.lab.proposer.debate import DebateProposer

        llm = build_llm(_llm_config(spec), **(llm_overrides or {}))
        n_per_persona = int(spec.proposer.extra.get("n_proposals_per_persona", 2))
        reflector = ThinkingChainReflector(
            max_generations=max_gens,
            n_proposals=n_per_persona,
        )
        proposer = DebateProposer(
            llm=llm,
            reflector=reflector,
            n_proposals_per_persona=n_per_persona,
        )
    elif ptype == "socratic":
        from siglab.lab.proposer.socratic import SocraticProposer

        llm = build_llm(_llm_config(spec), **(llm_overrides or {}))
        m_r1 = int(spec.proposer.extra.get("m_per_peer_r1", 2))
        reflector = ThinkingChainReflector(max_generations=max_gens, n_proposals=m_r1)
        proposer = SocraticProposer(
            llm=llm,
            reflector=reflector,
            k_per_peer=int(spec.proposer.extra.get("k_per_peer", 2)),
            m_per_peer_r1=m_r1,
        )
    else:
        raise ValueError(f"Unknown proposer type: {ptype!r}")

    evaluator = SignalAgentAdapter(
        agent=agent,
        exclude_financials=spec.evaluator.exclude_financials,
        exclude_microcap=spec.evaluator.exclude_microcap,
        alpha_factor_model=spec.evaluator.alpha_factor_model,
    )
    budget = Budget(
        max_tokens_in=spec.budget.max_tokens_in,
        max_tokens_out=spec.budget.max_tokens_out,
        max_wall_clock_seconds=spec.budget.max_wall_seconds,
    )
    out_dir = Path(output_dir_override or spec.experiment_folder("run"))

    return DiscoveryLoop(
        proposer=proposer,
        evaluator=evaluator,
        stopping=FixedGens(n=spec.loop.n_generations),
        budget=budget,
        recorder=Recorder(
            output_dir=out_dir,
            pair_name=spec.task.pair,
            run_name=spec.run.name,
        ),
        validator=ProposalValidator(
            reject_invalid_expression=bool(
                spec.loop.validation.get("reject_invalid_expression", True)
            ),
            reject_single_theme=bool(
                spec.loop.validation.get("reject_single_theme", True)
            ),
        ),
    )


def run_experiment(
    spec: RunSpec,
    *,
    agent: Any,
    timestamp: str,
    output_dir_override: str | Path | None = None,
    evaluator_wrapper: Callable[[Any], Any] | None = None,
) -> ExperimentRunResult:
    """Run one resolved RunSpec and write the canonical experiment artifacts."""
    task = build_task(spec)
    output_dir = Path(output_dir_override) if output_dir_override else experiment_dir(
        spec, timestamp,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    loop = build_components(
        spec,
        agent=agent,
        output_dir_override=output_dir,
    )
    if evaluator_wrapper is not None:
        loop.evaluator = evaluator_wrapper(loop.evaluator)

    loop.recorder.save_manifest(
        build_manifest(
            spec=spec,
            task=task,
            experiment_dir=output_dir,
            timestamp=timestamp,
        )
    )
    loop.recorder.save_resolved_config(
        resolved_config_payload(
            spec=spec,
            task=task,
            experiment_dir=output_dir,
        )
    )
    loop.recorder.save_task(task)

    archive = loop.run(
        task=task,
        n_proposals=spec.loop.proposals_per_generation,
    )
    write_search_universe(output_dir)
    write_posthoc_if_enabled(spec, output_dir)

    return ExperimentRunResult(
        spec=spec,
        task=task,
        archive=archive,
        loop=loop,
        output_dir=output_dir,
    )


def build_agent_from_env():
    """Build a siglab.agent.SignalEvaluator from cached CRSP/Compustat panels."""
    import pandas as pd

    from siglab.agent.executor import SignalEngine
    from siglab.agent.signal_evaluator import SignalEvaluator
    from siglab.data import cache, crsp as crsp_mod
    from siglab.data.factors import download_ff_factors

    cache.set_cache_dir(REPO_ROOT / "data" / "cache")
    crsp_panels = crsp_mod.get_crsp_panels(use_cache=True)

    crsp_raw = cache.load_panel("crsp_1960_2024_raw")
    crsp_raw["date"] = pd.to_datetime(crsp_raw["date"])
    crsp_raw["month"] = crsp_raw["date"].dt.month
    crsp_raw["year"] = crsp_raw["date"].dt.year

    comp = cache.load_panel("compustat_annual_1960_2024")
    ccm = cache.load_panel("ccm_link_1960_2024")
    comp_ccm = comp.merge(
        ccm[["gvkey", "permno", "linkdt", "linkenddt"]],
        on="gvkey",
        how="inner",
    )
    comp_ccm = comp_ccm[
        (comp_ccm["datadate"] >= comp_ccm["linkdt"])
        & (comp_ccm["datadate"] <= comp_ccm["linkenddt"])
    ]
    comp_ccm = comp_ccm.drop(columns=["linkdt", "linkenddt"])
    comp_ccm["year"] = comp_ccm["datadate"].dt.year
    comp_ccm = comp_ccm.sort_values(["permno", "year", "datadate"])
    comp_ccm = comp_ccm.drop_duplicates(subset=["permno", "year"], keep="last")
    comp_ccm["form_year"] = comp_ccm["year"] + 1

    sic_panel = None
    fin_panel = None
    if "siccd" in crsp_raw.columns:
        sic_panel = crsp_raw.pivot_table(
            values="siccd",
            index="date",
            columns="permno",
            aggfunc="first",
        )
        sic_panel.index = pd.to_datetime(sic_panel.index)
        fin_panel = (sic_panel >= 6000) & (sic_panel < 7000)

    ff_factors = download_ff_factors(use_cache=True)
    engine = SignalEngine(comp=comp_ccm, crsp=crsp_raw, sic_panel=sic_panel)
    return SignalEvaluator(
        engine=engine,
        ret_panel=crsp_panels["ret"],
        me_panel=crsp_panels["me"],
        ff_factors=ff_factors,
        nyse_panel=crsp_panels["nyse"],
        fin_panel=fin_panel,
    )


def write_posthoc_if_enabled(spec: RunSpec, exp_dir: Path) -> Path | None:
    if not spec.posthoc.enabled:
        return None
    return write_posthoc_manifest(
        experiment_dir=exp_dir,
        suites=spec.posthoc.suites,
    )


def write_search_universe(exp_dir: str | Path) -> Path:
    exp_dir = Path(exp_dir)
    return write_search_universe_summary(
        [exp_dir],
        exp_dir / "search_universe.json",
    )
