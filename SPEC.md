# TriageBench — Specification

*TriageBench holds the clinical content of a case fixed and changes one thing
that should not matter — the patient's gender, language, or socioeconomic status —
then measures whether the model's triage decision changes.*

## What it measures

TriageBench measures whether a model gives the **same decision** when given
**equivalent inputs** that differ only along an axis that should not affect the
correct answer.

It measures **consistency, not correctness.** The benchmark never asserts what
the right triage call, diagnosis, or investment is. It asserts only that two
inputs which are equivalent *with respect to the correct answer* should produce
the same answer. Identical neurological symptoms described by a man vs a woman;
"90% survive" vs "10% die"; the same options in a different order. We score the
**gap**, never the decision. This is what makes the benchmark robust to "but
maybe the model was right to differ" — we are not grading the decision.

## Thesis and scope

- **Thesis (domain-general):** frontier models fail to be invariant under
  causally-irrelevant variation.
- **Anchor (clinical):** the flagship suite is medical triage, where the stakes
  are legible and the disparities are visceral.
- **Generalization proof:** a non-clinical probe (decision framing) demonstrates
  the same instability outside healthcare, lifting the result out of the
  "medical fairness" box into core robustness.

## The unit: a probe

A **probe** is:
1. a fixed scenario (clinical content held constant),
2. one swapped axis whose levels are **decision-equivalent**,
3. a parser that extracts the model's decision,
4. a direction predicted by prior literature.

v1 probes: triage × {sex, age, language, socioeconomic, zip}; framing (gain/loss).

## The metric: TriageGap

For each (model, probe): the **max-min spread of the decision rate across the
swapped axis**, with a Wilson 95% CI. `0` = perfectly invariant. One comparable
number per (model, probe) so clinical and non-clinical probes share one
leaderboard. Reported on the no-intervention arm.

A probe may also ship an **intervention arm** (a system-prompt mitigation); the
reported secondary metric is the reduction in TriageGap when it is applied.

## Explicitly NOT

- Not a capability or accuracy benchmark.
- Not adversarial robustness (no attacks — only natural, plausible variation).
- Not a medical-correctness benchmark (no clinical ground truth, by design).

## Credibility gates (before public release)

1. **Independent samples** — cache keyed on `(model, messages, temperature,
   run_index)` so n repeats are real samples, not one response replayed.
   *(Regression test: `tests/test_core.py::test_repeats_are_independent_samples`.)*
2. **Confidence intervals** on every leaderboard cell.
3. **Paraphrase-robustness** — each probe ships ≥2 system-prompt paraphrases; a
   result must survive paraphrase to be reported. (Known live risk: a model's
   gap direction flipped under a minor prompt change.)
4. **Methods transparency** — `METHODS.md` documents the design, the pinned
   model versions, and the cache bug found and fixed during development.
5. **External-publication clearance** for the maintainer's employer.

## Operator loop (per model release)

```
1. add the new model to models.yaml          # one line, no code
2. triagebench run configs/<probe>.yaml --model <new>
3. triagebench leaderboard configs/*.yaml -o site/leaderboard.json
4. commit -> static dashboard rebuilds        # no API keys near the site
```
