from __future__ import annotations

from ..common import run_multiseed_pipeline as run_common_multiseed_pipeline
from ..common import run_single_seed_pipeline as run_common_single_seed_pipeline
from .engine import run_mogoa_engine


def run_mogoa_pipeline(**kwargs):
    return run_common_single_seed_pipeline("mogoa", run_mogoa_engine, **kwargs)


def run_mogoa_multiseed_pipeline(**kwargs):
    return run_common_multiseed_pipeline("mogoa", run_mogoa_pipeline, **kwargs)


def run_pipeline(**kwargs):
    return run_mogoa_pipeline(**kwargs)


def run_multiseed_pipeline(**kwargs):
    return run_mogoa_multiseed_pipeline(**kwargs)