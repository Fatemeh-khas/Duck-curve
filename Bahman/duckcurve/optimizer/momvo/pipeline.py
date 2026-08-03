from __future__ import annotations

from ..common import run_multiseed_pipeline as run_common_multiseed_pipeline
from ..common import run_single_seed_pipeline as run_common_single_seed_pipeline
from .engine import run_momvo_engine


def run_momvo_pipeline(**kwargs):
    return run_common_single_seed_pipeline("momvo", run_momvo_engine, **kwargs)


def run_momvo_multiseed_pipeline(**kwargs):
    return run_common_multiseed_pipeline("momvo", run_momvo_pipeline, **kwargs)


def run_pipeline(**kwargs):
    return run_momvo_pipeline(**kwargs)


def run_multiseed_pipeline(**kwargs):
    return run_momvo_multiseed_pipeline(**kwargs)