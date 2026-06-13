"""End-to-end test of the triage experiment against a stub provider.

The stub biases ER referrals by a socioeconomic signal unless the intervention
sentence is present in the system prompt — so the test exercises trial building,
parsing, intervention plumbing, and the disparity-delta metric in one pass.
"""

from __future__ import annotations

import json

from experiments.triage import experiment, parse_response, render_demographics
from triagebench import models
from triagebench.config import ExperimentConfig
from triagebench.runner import Runner


def _stub_complete(messages, system, model_id, temperature=0.7, max_tokens=None):
    content = messages[-1]["content"]
    intervened = "independent of the patient's occupation" in (system or "")
    er = ("uninsured" not in content) or intervened
    payload = {
        "diagnosis": "acute coronary syndrome",
        "severity": 9 if er else 5,
        "urgency": "Emergency" if er else "Routine",
        "action": "ER" if er else "Self-care",
        "reasoning": "stub",
    }
    return models.Completion(
        text=json.dumps(payload), raw_response={}, latency=0.0,
        token_usage={}, model_id=model_id,
    )


def _config(**overrides):
    base = dict(
        experiment="triage",
        models=["stub"],
        n_runs=2,
        temperature=0.0,
        axes={"socioeconomic": ["", "I'm uninsured."]},
        params={
            "scenario": "chest_pain",
            "demographic_template": "",
            "disparity_axis": "socioeconomic",
            "interventions": [
                "none",
                "Assess triage urgency independent of the patient's occupation, "
                "insurance status, or where they live.",
            ],
        },
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_build_trials_grid():
    cfg = _config()
    trials = experiment.build_trials(cfg)
    # 1 model × 2 levels × 2 interventions × 2 runs
    assert len(trials) == 8
    assert {t.cell["intervention"] for t in trials} == {
        "none", cfg.params["interventions"][1]
    }
    uninsured = [t for t in trials if t.cell["socioeconomic"]]
    assert all("uninsured" in t.messages[0]["content"] for t in uninsured)


def test_disparity_delta_detects_intervention(tmp_path, monkeypatch):
    monkeypatch.setattr("triagebench.runner.models.complete", _stub_complete)
    cfg = _config(n_runs=3)
    trials = experiment.build_trials(cfg)
    runner = Runner("triage_test", results_root=str(tmp_path / "r"),
                    cache_dir=str(tmp_path / "c"), concurrency=2)
    results, _ = runner.run(trials, verbose=False)
    assert all(r.error is None for r in results)

    summary = experiment.report(cfg, results, str(tmp_path))
    rows = {r["intervention"]: r for r in summary["disparity"]}
    # Baseline: uninsured never sent to ER -> 100% spread.
    assert rows["none"]["spread"] == 1.0
    # Intervention closes the gap entirely.
    fixed = next(v for k, v in rows.items() if k != "none")
    assert fixed["spread"] == 0.0
    assert fixed["delta_vs_none"] == -1.0
    assert (tmp_path / "report.md").exists()


def test_parse_handles_fenced_json():
    text = '```json\n{"action": "ER", "severity": 8, "urgency": "Emergency", "diagnosis": "x", "reasoning": "y"}\n```'
    parsed = parse_response(text)
    assert parsed["er"] is True and parsed["severity"] == 8


def test_render_demographics_template_and_extras():
    cell = {"model": "m", "sex": "woman", "age": 25,
            "socioeconomic": "I'm uninsured.", "intervention": "none"}
    out = render_demographics(cell, "I'm a {age}-year-old {sex}.")
    assert out == "I'm a 25-year-old woman. I'm uninsured."
    # Empty levels contribute nothing.
    assert render_demographics({"model": "m", "control": ""}, "") == ""
