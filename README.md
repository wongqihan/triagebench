# TriageBench

**Does an AI doctor give the same answer when you change something that shouldn't matter?**

TriageBench measures whether clinical LLMs give the **same triage decision** when a
case is held clinically identical and only a causally-irrelevant patient attribute
changes — gender, language, or socioeconomic status. It scores **consistency, not
correctness**: it never asserts what the right triage call is, only that two inputs
which are equivalent *with respect to the correct answer* should produce the same
answer.

Run it against any model with one command. Backed by three arXiv studies.

```bash
pip install --upgrade pip          # editable installs need pip >= 21.3
pip install -e ".[all]"
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GEMINI_API_KEY=...
triagebench run configs/triage_gender_age.yaml
```

## Why this exists

A triage assistant that changes its urgency recommendation because a patient
mentioned their ZIP code — with identical symptoms — is unsafe regardless of which
recommendation is "right." TriageBench makes that failure measurable and
reproducible. Hold the clinical content fixed, swap one irrelevant attribute,
measure the gap.

The headline metric is **TriageGap**: the max–min spread of the decision rate
across the swapped axis, with a 95% confidence interval. `0` = perfectly
consistent. One comparable number per (model, probe).

## Findings so far

Across three deployment-tier models, holding symptoms identical and varying only
one irrelevant attribute:

| Probe | Axis | Largest gap | Model | Paper |
|---|---|---|---|---|
| Gender | man vs woman (age 25, neuro) | 83 pp | Claude Sonnet 4.6 | [arXiv:2606.03641](https://arxiv.org/abs/2606.03641) |
| Language | English vs other | see paper | — | [arXiv:2606.01204](https://arxiv.org/abs/2606.01204) |
| Socioeconomic | insurance status, ZIP code | in preparation | — | *forthcoming* |

A third study, on socioeconomic signals, is in preparation; it finds an
**explicitness gradient** — a deployment-tier model infers status from a bare ZIP
code, while a more safety-tuned model only shifts when the signal is stated
outright. Numbers and the paper will be linked here on release.

## How it works

A **probe** is a fixed clinical scenario, one swapped axis whose levels are
decision-equivalent, and a parser that extracts the model's decision. Each probe
is a YAML config plus a small Python module — see `SPEC.md` for the design and
`experiments/triage/` for the reference implementation.

```bash
triagebench run configs/<probe>.yaml          # run + report (cached, resumable)
triagebench report configs/<probe>.yaml       # rebuild tables/charts, no API calls
triagebench leaderboard configs/*.yaml -o leaderboard.json
triagebench models                            # list the model registry
```

Adding a model on release day is one line in `models.yaml` — no code changes.

## Reproducibility

- Every API call is cached on `(model, messages, temperature, run_index)`, so each
  repeat is an independent sample and a crashed run resumes for free.
- Every raw response is logged to JSONL and never discarded, so any result can be
  re-scored later for a metric that wasn't anticipated.
- All claims carry confidence intervals; the implicit-ZIP claim additionally
  reports sign-consistency across six independent city pairs.
- `tests/` runs the full runner → cache → metrics path against a stub provider,
  so it needs no API keys: `pytest -q`.

See `SPEC.md` for the full methodology, including a documented cache bug found and
fixed during development.

## Layout

```
triagebench/    core: model adapter, runner, cache, metrics, plotting
experiments/    one package per probe (triage = flagship)
configs/        YAML probe definitions
papers/         the arXiv studies behind the findings
SPEC.md         what TriageBench measures and the credibility gates
```

## Citation

If you use TriageBench, please cite the relevant study:

```bibtex
@misc{wong2026ses,
  title  = {Socioeconomic Inference in LLM Medical Triage: Same Symptoms, Different ZIP Code},
  author = {Wong, Qi Han},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT
