from __future__ import annotations

from ..common import run_multiseed_pipeline as run_common_multiseed_pipeline
from ..common import run_single_seed_pipeline as run_common_single_seed_pipeline
from .engine import run_gde3_engine


def run_gde3_pipeline(**kwargs):
    return run_common_single_seed_pipeline("gde3", run_gde3_engine, **kwargs)


def run_gde3_multiseed_pipeline(**kwargs):
    return run_common_multiseed_pipeline("gde3", run_gde3_pipeline, **kwargs)


def run_pipeline(**kwargs):
    return run_gde3_pipeline(**kwargs)


def run_multiseed_pipeline(**kwargs):
    return run_gde3_multiseed_pipeline(**kwargs)