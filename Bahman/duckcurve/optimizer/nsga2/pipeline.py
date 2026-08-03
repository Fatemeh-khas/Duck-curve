from __future__ import annotations

from ..common import run_multiseed_pipeline as run_common_multiseed_pipeline
from ..common import run_single_seed_pipeline as run_common_single_seed_pipeline
from .engine import run_nsga2_engine


def run_nsga2_pipeline(**kwargs):
    return run_common_single_seed_pipeline("nsga2", run_nsga2_engine, **kwargs)


def run_nsga2_multiseed_pipeline(**kwargs):
    return run_common_multiseed_pipeline("nsga2", run_nsga2_pipeline, **kwargs)


def run_pipeline(**kwargs):
    return run_nsga2_pipeline(**kwargs)


def run_multiseed_pipeline(**kwargs):
    return run_nsga2_multiseed_pipeline(**kwargs)