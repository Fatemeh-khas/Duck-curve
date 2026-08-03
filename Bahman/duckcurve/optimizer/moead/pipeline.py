from __future__ import annotations

from ..common import run_multiseed_pipeline as run_common_multiseed_pipeline
from ..common import run_single_seed_pipeline as run_common_single_seed_pipeline
from .engine import run_moead_engine


def run_moead_pipeline(**kwargs):
    return run_common_single_seed_pipeline("moead", run_moead_engine, **kwargs)


def run_moead_multiseed_pipeline(**kwargs):
    return run_common_multiseed_pipeline("moead", run_moead_pipeline, **kwargs)


def run_pipeline(**kwargs):
    return run_moead_pipeline(**kwargs)


def run_multiseed_pipeline(**kwargs):
    return run_moead_multiseed_pipeline(**kwargs)