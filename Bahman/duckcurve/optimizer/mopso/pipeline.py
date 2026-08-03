from __future__ import annotations

from ..common import run_multiseed_pipeline as run_common_multiseed_pipeline
from ..common import run_single_seed_pipeline as run_common_single_seed_pipeline
from .engine import run_mopso_engine


def run_mopso_pipeline(**kwargs):
    return run_common_single_seed_pipeline("mopso", run_mopso_engine, **kwargs)


def run_mopso_multiseed_pipeline(**kwargs):
    return run_common_multiseed_pipeline("mopso", run_mopso_pipeline, **kwargs)


def run_pipeline(**kwargs):
    return run_mopso_pipeline(**kwargs)


def run_multiseed_pipeline(**kwargs):
    return run_mopso_multiseed_pipeline(**kwargs)