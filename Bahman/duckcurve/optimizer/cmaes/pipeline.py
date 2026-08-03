from __future__ import annotations

from ..common import run_multiseed_pipeline as run_common_multiseed_pipeline
from ..common import run_single_seed_pipeline as run_common_single_seed_pipeline
from .engine import run_sep_cmaes_engine


def run_cmaes_pipeline(**kwargs):
    return run_common_single_seed_pipeline("cmaes", run_sep_cmaes_engine, **kwargs)


def run_cmaes_multiseed_pipeline(**kwargs):
    return run_common_multiseed_pipeline("cmaes", run_cmaes_pipeline, **kwargs)


def run_pipeline(**kwargs):
    return run_cmaes_pipeline(**kwargs)


def run_multiseed_pipeline(**kwargs):
    return run_cmaes_multiseed_pipeline(**kwargs)