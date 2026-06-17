"""Experiment config loading + validation (pydantic).

A config is a YAML file naming the experiment module, the models, the
independent-variable grid (cartesian product = cells), n_runs per cell,
temperature, and any experiment-specific params.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    experiment: str = Field(..., description="Name of the experiment module under experiments/")
    models: List[str] = Field(..., min_length=1)
    n_runs: int = Field(1, ge=1)
    # TriageBench standard temperature. Low enough for tight, reproducible CIs,
    # high enough that decision rates stay graded (temperature 0 collapses rates
    # to 0/100 and degenerates the gap metric). Configs may override.
    temperature: float = Field(0.3, ge=0.0, le=2.0)

    # The independent-variable grid. Each key is an axis; each value is a list of
    # levels. The cartesian product (with model) defines the cells.
    axes: Dict[str, List[Any]] = Field(default_factory=dict)

    # Anything experiment-specific (scenario name, checkpoints, interventions, …).
    params: Dict[str, Any] = Field(default_factory=dict)

    # Runner knobs.
    concurrency: int = Field(4, ge=1)
    use_cache: bool = True

    @field_validator("models")
    @classmethod
    def _no_dupes(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("duplicate model ids in `models`")
        return v

    def cells(self) -> List[Dict[str, Any]]:
        """Cartesian product of model × every axis = the list of cells."""
        axis_names = list(self.axes)
        level_lists = [self.axes[a] for a in axis_names]
        cells = []
        for model in self.models:
            if axis_names:
                for combo in itertools.product(*level_lists):
                    cell = {"model": model}
                    cell.update(dict(zip(axis_names, combo)))
                    cells.append(cell)
            else:
                cells.append({"model": model})
        return cells


def load_config(path: str) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return ExperimentConfig(**raw)
