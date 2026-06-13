"""Trial / TrialResult dataclasses — the unit of work in the harness."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# A parse callable takes the model's response text and returns the dependent
# variables for this trial as a flat dict (e.g. {"action": "ER", "severity": 7}).
ParseFn = Callable[[str], Dict[str, Any]]


@dataclass
class Trial:
    """A single unit of work: one prompt sent to one model, parsed one way.

    Attributes:
        id: Stable identifier (unique within a run).
        cell: The independent-variable values that define this trial's group,
            e.g. {"model": "claude", "pressure": "emotional"}. Trials sharing a
            cell are aggregated together by metrics.
        messages: Chat messages to send (list of {"role", "content"}).
        system: Optional system prompt.
        model_id: Registry key naming the model to dispatch to.
        temperature: Sampling temperature.
        parse: Callable mapping response text -> dependent-variable dict.
        run_index: Which repeat this is within the cell (0..n_runs-1).
        meta: Anything else the experiment wants to carry through to results.
    """

    id: str
    cell: Dict[str, Any]
    messages: List[Dict[str, str]]
    model_id: str
    parse: ParseFn
    system: Optional[str] = None
    temperature: float = 0.7
    run_index: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    """The outcome of running a Trial.

    The raw response text is ALWAYS retained so results can be re-parsed later
    for metrics that weren't anticipated at run time. Reasoning/thinking content
    is stored in its own field so the decision is never contaminated by the
    explanation.
    """

    trial_id: str
    cell: Dict[str, Any]
    parsed: Dict[str, Any]
    raw_text: str
    reasoning: Optional[str] = None
    probe_reasoning: Optional[str] = None
    model_id: str = ""
    latency: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    cached: bool = False
    error: Optional[str] = None
    run_index: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> Dict[str, Any]:
        """Flatten to a JSON-serializable dict for one JSONL line."""
        return {
            "trial_id": self.trial_id,
            "cell": self.cell,
            "parsed": self.parsed,
            "raw_text": self.raw_text,
            "reasoning": self.reasoning,
            "probe_reasoning": self.probe_reasoning,
            "model_id": self.model_id,
            "latency": round(self.latency, 4),
            "token_usage": self.token_usage,
            "cached": self.cached,
            "error": self.error,
            "run_index": self.run_index,
            "meta": self.meta,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "TrialResult":
        return cls(
            trial_id=d["trial_id"],
            cell=d["cell"],
            parsed=d.get("parsed", {}),
            raw_text=d.get("raw_text", ""),
            reasoning=d.get("reasoning"),
            probe_reasoning=d.get("probe_reasoning"),
            model_id=d.get("model_id", ""),
            latency=d.get("latency", 0.0),
            token_usage=d.get("token_usage", {}),
            cached=d.get("cached", False),
            error=d.get("error"),
            run_index=d.get("run_index", 0),
            meta=d.get("meta", {}),
            timestamp=d.get("timestamp", 0.0),
        )
