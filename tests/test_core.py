"""End-to-end smoke test of the core harness using a stub model provider.

Runs with no API keys: we register a fake 'stub' model and monkeypatch the
provider dispatch so the runner, cache, JSONL logging, and metrics all exercise
real code paths against deterministic output.
"""

from __future__ import annotations

import json

from triagebench import metrics, models
from triagebench.config import ExperimentConfig
from triagebench.runner import Runner
from triagebench.trial import Trial


def _stub_complete(messages, system, model_id, temperature=0.7, max_tokens=None):
    # Echo the demographic slot back as a fake JSON triage answer.
    content = messages[-1]["content"]
    sev = 9 if "a man" in content else 6
    text = json.dumps({"action": "ER" if sev >= 8 else "Self-care", "severity": sev})
    return models.Completion(
        text=text, raw_response={"stub": True}, latency=0.01,
        token_usage={"input_tokens": 5, "output_tokens": 5}, model_id=model_id,
    )


def _parse(text):
    return json.loads(text)


def test_run_cache_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "complete", _stub_complete)
    monkeypatch.setattr("triagebench.runner.models.complete", _stub_complete)

    trials = [
        Trial(id=f"t{i}", cell={"model": "stub", "sex": sex},
              messages=[{"role": "user", "content": f"I'm a {sex}. Headache."}],
              model_id="stub", parse=_parse, run_index=i)
        for sex in ("man", "woman") for i in range(3)
    ]

    runner = Runner("unit_test", results_root=str(tmp_path / "r"),
                    cache_dir=str(tmp_path / "c"), concurrency=2)
    results, path = runner.run(trials, verbose=False)

    assert len(results) == 6
    assert all(r.error is None for r in results)
    assert path.exists() and len(path.read_text().strip().splitlines()) == 6

    # Cache hit on re-run: every result should be cached the second time.
    results2, _ = runner.run(trials, verbose=False)
    assert all(r.cached for r in results2)

    # Metrics: ER rate should differ by sex (9 vs 6 severity -> ER vs not).
    grouped = metrics.group_by_cell(results, axes=["sex"])
    by_sex = {dict(k)["sex"]: metrics.proportion(v, "action", "ER")[0]
              for k, v in grouped.items()}
    assert by_sex["man"] == 1.0
    assert by_sex["woman"] == 0.0


def test_config_cells():
    cfg = ExperimentConfig(
        experiment="triage", models=["claude", "gpt"], n_runs=2,
        axes={"sex": ["m", "f"], "age": [25, 65]},
    )
    # 2 models × 2 sex × 2 age = 8 cells
    assert len(cfg.cells()) == 8


def test_repeats_are_independent_samples(tmp_path, monkeypatch):
    """Regression: n_runs repeats of one cell must each hit the API, not replay
    the first response from cache (cache key must include run_index)."""
    calls = {"n": 0}

    def counting_stub(messages, system, model_id, temperature=0.7, max_tokens=None):
        calls["n"] += 1
        return models.Completion(
            text=json.dumps({"action": "ER", "severity": calls["n"]}),
            raw_response={}, latency=0.0, token_usage={}, model_id=model_id,
        )

    monkeypatch.setattr("triagebench.runner.models.complete", counting_stub)
    trials = [
        Trial(id=f"r{i}", cell={"model": "stub"}, run_index=i,
              messages=[{"role": "user", "content": "same prompt every time"}],
              model_id="stub", parse=_parse)
        for i in range(10)
    ]
    runner = Runner("dup_test", results_root=str(tmp_path / "r"),
                    cache_dir=str(tmp_path / "c"), concurrency=1)
    results, _ = runner.run(trials, verbose=False)
    assert calls["n"] == 10, "identical prompts with distinct run_index must not share cache"
    assert len({r.parsed["severity"] for r in results}) == 10
