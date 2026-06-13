"""The experiment-module contract.

Each experiment under experiments/<name>/ exposes a module-level object (or the
package itself) implementing this protocol. The CLI imports it by the config's
`experiment` name and calls these hooks. Keeping the surface this small is what
lets a whole experiment be "a YAML file plus a small Python module."
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from .config import ExperimentConfig
from .trial import Trial, TrialResult


@runtime_checkable
class Experiment(Protocol):
    name: str

    def build_trials(self, config: ExperimentConfig) -> List[Trial]:
        """Expand the config's cells × n_runs into concrete Trials."""
        ...

    def report(
        self, config: ExperimentConfig, results: List[TrialResult], out_dir: str
    ) -> Dict[str, Any]:
        """Reduce results into metrics, write tables/charts under out_dir, and
        return a JSON-serializable summary (also used by the `report` command)."""
        ...


def load_experiment(name: str) -> "Experiment":
    """Import experiments.<name> and return its module-level `experiment`."""
    import importlib

    mod = importlib.import_module(f"experiments.{name}")
    exp = getattr(mod, "experiment", None)
    if exp is None:
        raise AttributeError(
            f"experiments.{name} must expose a module-level `experiment` object"
        )
    return exp
