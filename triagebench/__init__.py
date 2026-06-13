"""experiment-harness: a config-driven LLM experiment harness.

One core harness, many experiments. An experiment is a YAML config plus a small
module that builds trials and reports metrics. The harness handles model
dispatch, concurrency, caching, retries, JSONL logging, and plotting.
"""

from .config import ExperimentConfig, load_config
from .metrics import get_metric, metric, registered, wilson_ci
from .models import Completion, complete
from .runner import Runner, load_results
from .trial import Trial, TrialResult

__all__ = [
    "ExperimentConfig",
    "load_config",
    "metric",
    "get_metric",
    "registered",
    "wilson_ci",
    "complete",
    "Completion",
    "Runner",
    "load_results",
    "Trial",
    "TrialResult",
]

__version__ = "0.1.0"
