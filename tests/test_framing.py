"""Framing experiment: trial building, parsing, and the framing-effect metric,
exercised end-to-end against a stub that is frame-sensitive only without the
equivalence note."""

from __future__ import annotations

import json

from experiments.framing import ITEMS, experiment, make_parse
from triagebench import models
from triagebench.config import ExperimentConfig
from triagebench.runner import Runner


def _stub_complete(messages, system, model_id, temperature=0.7, max_tokens=None):
    content = messages[-1]["content"]
    item_key = next(k for k, v in ITEMS.items() if v["question"] in content)
    item = ITEMS[item_key]
    is_gain = item["gain"] in content
    debiased = "numerically identical" in (system or "")
    # Frame-sensitive unless debiased: target under gain, alternative under loss.
    if debiased or is_gain:
        choice = item["target"]
    else:
        choice = next(o for o in item["options"] if o != item["target"])
    payload = {"choice": choice, "favorability": 8 if is_gain else 4, "reasoning": "stub"}
    return models.Completion(
        text=json.dumps(payload), raw_response={}, latency=0.0,
        token_usage={}, model_id=model_id,
    )


def _config(n_runs=2):
    return ExperimentConfig(
        experiment="framing",
        models=["stub"],
        n_runs=n_runs,
        temperature=0.0,
        axes={"frame": ["gain", "loss"], "item": list(ITEMS)},
        params={"interventions": ["none", "Note: ... numerically identical ..."]},
    )


def test_build_trials_grid():
    trials = experiment.build_trials(_config())
    # 1 model x 2 frames x 5 items x 2 arms x 2 runs
    assert len(trials) == 40
    prompts = {t.messages[0]["content"] for t in trials}
    assert len(prompts) == 10  # 5 items x 2 frames


def test_parse_matches_options_loosely():
    parse = make_parse("policy_disease")
    out = parse(json.dumps({"choice": "program a", "favorability": 7, "reasoning": "x"}))
    assert out["choice"] == "Program A" and out["chose_target"] is True


def test_framing_effect_metric(tmp_path, monkeypatch):
    monkeypatch.setattr("triagebench.runner.models.complete", _stub_complete)
    cfg = _config(n_runs=3)
    trials = experiment.build_trials(cfg)
    runner = Runner("framing_test", results_root=str(tmp_path / "r"),
                    cache_dir=str(tmp_path / "c"), concurrency=2)
    results, _ = runner.run(trials, verbose=False)
    assert all(r.error is None for r in results)

    summary = experiment.report(cfg, results, str(tmp_path))
    rows = {(r["item"], r["intervention"]): r for r in summary["framing_effects"]}
    # Stub is fully frame-driven without the note...
    assert rows[("medical_surgery", "none")]["framing_effect"] == 1.0
    # ...and frame-invariant with it.
    assert rows[("medical_surgery", "equivalence_note")]["framing_effect"] == 0.0
    # Favorability shift is captured.
    assert rows[("medical_surgery", "none")]["favorability_shift"] == 4.0
    assert (tmp_path / "report.md").exists()
