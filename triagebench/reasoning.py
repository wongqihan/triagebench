"""Reasoning capture utilities.

Two things live here:
1. Helpers for treating provider-exposed thinking content as a first-class output
   (already captured on Completion.reasoning by models.py).
2. A "reasoning probe" mode: after a decision turn, append a follow-up turn that
   asks the model to explain its decision. The explanation is stored SEPARATELY
   (TrialResult.probe_reasoning) so the original decision is never contaminated by
   the act of asking for a justification.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import models

DEFAULT_PROBE = (
    "Briefly explain the main factors behind the recommendation you just gave. "
    "Do not change your recommendation."
)


def probe_decision(
    messages: List[Dict[str, str]],
    system: Optional[str],
    assistant_reply: str,
    model_id: str,
    temperature: float = 0.7,
    probe_prompt: str = DEFAULT_PROBE,
) -> str:
    """Run a follow-up turn asking the model to justify its prior answer.

    `assistant_reply` is the decision text from the first turn; it is appended as
    an assistant message so the model explains the answer it already committed to.
    Returns the explanation text only — the caller stores it in probe_reasoning.
    """
    followup = list(messages) + [
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": probe_prompt},
    ]
    comp = models.complete(
        messages=followup,
        system=system,
        model_id=model_id,
        temperature=temperature,
    )
    return comp.text


def split_reasoning(text: str, markers=("reasoning", "rationale", "because")) -> Dict[str, str]:
    """Best-effort split of an inline 'decision … reasoning …' blob.

    Used only when a model doesn't expose structured thinking and the experiment
    didn't force a JSON reasoning field. Prefer structured output where possible.
    """
    low = text.lower()
    for m in markers:
        idx = low.find(m)
        if idx > 0:
            return {"decision": text[:idx].strip(), "reasoning": text[idx:].strip()}
    return {"decision": text.strip(), "reasoning": ""}
