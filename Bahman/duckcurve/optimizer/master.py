from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .common import OptimizerPipelineResult
from .cmaes import run_multiseed_pipeline as run_cmaes_multiseed_pipeline, run_pipeline as run_cmaes_pipeline
from .ezoa import run_multiseed_pipeline as run_ezoa_multiseed_pipeline, run_pipeline as run_ezoa_pipeline
from .gde3 import run_multiseed_pipeline as run_gde3_multiseed_pipeline, run_pipeline as run_gde3_pipeline
from .moalo import run_multiseed_pipeline as run_moalo_multiseed_pipeline, run_pipeline as run_moalo_pipeline
from .moead import run_multiseed_pipeline as run_moead_multiseed_pipeline, run_pipeline as run_moead_pipeline
from .moda import run_multiseed_pipeline as run_moda_multiseed_pipeline, run_pipeline as run_moda_pipeline
from .modhl import run_multiseed_pipeline as run_modhl_multiseed_pipeline, run_pipeline as run_modhl_pipeline
from .mogoa import run_multiseed_pipeline as run_mogoa_multiseed_pipeline, run_pipeline as run_mogoa_pipeline
from .mogwo import run_multiseed_pipeline as run_mogwo_multiseed_pipeline, run_pipeline as run_mogwo_pipeline
from .momvo import run_multiseed_pipeline as run_momvo_multiseed_pipeline, run_pipeline as run_momvo_pipeline
from .mopso import run_multiseed_pipeline as run_mopso_multiseed_pipeline, run_pipeline as run_mopso_pipeline
from .mowoa import run_multiseed_pipeline as run_mowoa_multiseed_pipeline, run_pipeline as run_mowoa_pipeline
from .nsga2 import run_multiseed_pipeline as run_nsga2_multiseed_pipeline, run_pipeline as run_nsga2_pipeline

DEFAULT_OPTIMIZER_NAME = "ezoa"


@dataclass(frozen=True)
class OptimizerAdapter:
    name: str
    run_pipeline: Callable[..., OptimizerPipelineResult]
    run_multiseed_pipeline: Callable[..., OptimizerPipelineResult]


_OPTIMIZER_ALIASES = {
    "ozoa": "ezoa",
    "de": "gde3",
    "mode": "gde3",
    "omopso": "mopso",
    "mo-cmaes": "cmaes",
    "sep-cmaes": "cmaes",
}

_OPTIMIZER_REGISTRY = {
    "cmaes": OptimizerAdapter(
        name="cmaes",
        run_pipeline=run_cmaes_pipeline,
        run_multiseed_pipeline=run_cmaes_multiseed_pipeline,
    ),
    "ezoa": OptimizerAdapter(
        name="ezoa",
        run_pipeline=run_ezoa_pipeline,
        run_multiseed_pipeline=run_ezoa_multiseed_pipeline,
    ),
    "gde3": OptimizerAdapter(
        name="gde3",
        run_pipeline=run_gde3_pipeline,
        run_multiseed_pipeline=run_gde3_multiseed_pipeline,
    ),
    "moalo": OptimizerAdapter(
        name="moalo",
        run_pipeline=run_moalo_pipeline,
        run_multiseed_pipeline=run_moalo_multiseed_pipeline,
    ),
    "moead": OptimizerAdapter(
        name="moead",
        run_pipeline=run_moead_pipeline,
        run_multiseed_pipeline=run_moead_multiseed_pipeline,
    ),
    "moda": OptimizerAdapter(
        name="moda",
        run_pipeline=run_moda_pipeline,
        run_multiseed_pipeline=run_moda_multiseed_pipeline,
    ),
    "modhl": OptimizerAdapter(
        name="modhl",
        run_pipeline=run_modhl_pipeline,
        run_multiseed_pipeline=run_modhl_multiseed_pipeline,
    ),
    "mogoa": OptimizerAdapter(
        name="mogoa",
        run_pipeline=run_mogoa_pipeline,
        run_multiseed_pipeline=run_mogoa_multiseed_pipeline,
    ),
    "mogwo": OptimizerAdapter(
        name="mogwo",
        run_pipeline=run_mogwo_pipeline,
        run_multiseed_pipeline=run_mogwo_multiseed_pipeline,
    ),
    "momvo": OptimizerAdapter(
        name="momvo",
        run_pipeline=run_momvo_pipeline,
        run_multiseed_pipeline=run_momvo_multiseed_pipeline,
    ),
    "mopso": OptimizerAdapter(
        name="mopso",
        run_pipeline=run_mopso_pipeline,
        run_multiseed_pipeline=run_mopso_multiseed_pipeline,
    ),
    "mowoa": OptimizerAdapter(
        name="mowoa",
        run_pipeline=run_mowoa_pipeline,
        run_multiseed_pipeline=run_mowoa_multiseed_pipeline,
    ),
    "nsga2": OptimizerAdapter(
        name="nsga2",
        run_pipeline=run_nsga2_pipeline,
        run_multiseed_pipeline=run_nsga2_multiseed_pipeline,
    ),
}


def normalize_optimizer_name(name: str) -> str:
    normalized = str(name).strip().lower()
    return _OPTIMIZER_ALIASES.get(normalized, normalized)


def available_optimizers() -> tuple[str, ...]:
    return tuple(sorted(_OPTIMIZER_REGISTRY))


def get_optimizer_adapter(name: str) -> OptimizerAdapter:
    normalized = normalize_optimizer_name(name)
    adapter = _OPTIMIZER_REGISTRY.get(normalized)
    if adapter is None:
        choices = ", ".join(available_optimizers())
        raise ValueError(f"Unknown optimizer '{name}'. Available optimizers: {choices}")
    return adapter


def run_optimizer_pipeline(name: str = DEFAULT_OPTIMIZER_NAME, **kwargs) -> OptimizerPipelineResult:
    return get_optimizer_adapter(name).run_pipeline(**kwargs)


def run_optimizer_multiseed_pipeline(
    name: str = DEFAULT_OPTIMIZER_NAME,
    **kwargs,
) -> OptimizerPipelineResult:
    return get_optimizer_adapter(name).run_multiseed_pipeline(**kwargs)