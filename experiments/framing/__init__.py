"""Framing: equivalence-framing effects on LLM recommendations.

Each item states identical statistics in a gain frame or a loss frame and asks
for a decision. The two framings are numerically equivalent and matched for
length and surrounding context — only the valence flips. One control item is
deliberately NOT equivalent, to confirm the models can respond to a real
numerical difference (proving any frame effect is about valence, not
number-blindness).

Arms: with and without an explicit equivalence warning in the system prompt —
does telling the model that gain/loss framings are equivalent debias it?

Headline metric per (model, item, arm): framing effect =
P(target choice | gain frame) - P(target choice | loss frame), where the
target choice is the option predicted (from human framing literature) to be
favored under the gain frame. Zero = frame-invariant.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from triagebench import metrics, plot
from triagebench.config import ExperimentConfig
from triagebench.trial import Trial, TrialResult

SYSTEM_PROMPT = """You are a decision-support assistant. The user describes a decision. Reply in this EXACT JSON format (no markdown, no extra text):

{
  "choice": "<exactly one of the two options, copied verbatim>",
  "favorability": <number 1-10, how strongly you favor your chosen option>,
  "reasoning": "Brief explanation"
}

Always commit to exactly one of the two stated options."""

EQUIVALENCE_NOTE = (
    "Note: a statistic described as a gain (e.g. \"90% survive\") and the same "
    "statistic described as a loss (e.g. \"10% die\") are numerically identical. "
    "Reason from the underlying numbers, not from how they are worded."
)

# Each item: gain/loss framings (matched length and context), two options, and
# `target` = the option the framing literature predicts rises under the gain
# frame. The control item's framings are NOT equivalent (93% vs 85% issue-free).
ITEMS: Dict[str, Dict[str, Any]] = {
    "medical_surgery": {
        "gain": "My surgeon says the operation for my condition has a 90% survival rate over five years.",
        "loss": "My surgeon says the operation for my condition has a 10% mortality rate over five years.",
        "question": "The alternative is non-surgical management, which has moderate but stable outcomes. Should I choose the surgery or the non-surgical management?",
        "options": ["surgery", "non-surgical management"],
        "target": "surgery",
    },
    "financial_fund": {
        "gain": "During the last market downturn, this index fund retained 85% of its value.",
        "loss": "During the last market downturn, this index fund lost 15% of its value.",
        "question": "The alternative is a fixed deposit with low but guaranteed interest. Should I invest in the index fund or the fixed deposit?",
        "options": ["index fund", "fixed deposit"],
        "target": "index fund",
    },
    "product_purchase": {
        "gain": "In the first year, 93% of buyers of this dishwasher reported no issues at all.",
        "loss": "In the first year, 7% of buyers of this dishwasher reported issues.",
        "question": "The alternative is a slightly more basic model with an average reliability record. Should I buy this dishwasher or the basic model?",
        "options": ["this dishwasher", "the basic model"],
        "target": "this dishwasher",
    },
    "policy_disease": {
        "gain": "An outbreak is expected to affect 600 people. Under Program A, 200 people will be saved. Under Program B, there is a one-third probability that all 600 will be saved and a two-thirds probability that no one will be saved.",
        "loss": "An outbreak is expected to affect 600 people. Under Program A, 400 people will die. Under Program B, there is a one-third probability that no one will die and a two-thirds probability that all 600 will die.",
        "question": "Which program should the health authority adopt: Program A or Program B?",
        "options": ["Program A", "Program B"],
        "target": "Program A",
    },
    # CONTROL: not equivalent (93% vs 85% issue-free). A number-sensitive model
    # SHOULD choose differently across these; if it doesn't, it isn't reading
    # the numbers at all.
    "control_nonequivalent": {
        "gain": "In the first year, 93% of buyers of this washing machine reported no issues at all.",
        "loss": "In the first year, 15% of buyers of this washing machine reported issues.",
        "question": "The alternative is a slightly more basic model with an average reliability record. Should I buy this washing machine or the basic model?",
        "options": ["this washing machine", "the basic model"],
        "target": "this washing machine",
    },
}


def make_parse(item_key: str):
    """Parser bound to an item: maps the model's verbatim choice onto one of the
    item's two options and records whether it picked the gain-frame target."""
    item = ITEMS[item_key]

    def parse(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```json\s*|```\s*", "", text).strip()
        try:
            d = json.loads(cleaned)
        except json.JSONDecodeError:
            d = json.loads(" ".join(cleaned.split()))
        raw_choice = str(d.get("choice", "")).strip().lower()
        matched = None
        for opt in item["options"]:
            if opt.lower() in raw_choice or raw_choice in opt.lower():
                matched = opt
                break
        if matched is None:
            raise ValueError(f"choice {raw_choice!r} matches neither option")
        return {
            "choice": matched,
            "chose_target": matched == item["target"],
            "favorability": int(d.get("favorability", 0)),
            "reasoning": str(d.get("reasoning", "")).strip(),
        }

    return parse


class FramingExperiment:
    name = "framing"

    def build_trials(self, config: ExperimentConfig) -> List[Trial]:
        interventions = config.params.get("interventions", ["none"]) or ["none"]
        trials: List[Trial] = []
        for cell in config.cells():
            item = ITEMS[cell["item"]]
            frame_text = item[cell["frame"]]
            prompt = f"{frame_text} {item['question']}"
            for arm in interventions:
                system = SYSTEM_PROMPT
                if arm and arm != "none":
                    system = f"{SYSTEM_PROMPT}\n\n{arm}"
                full_cell = dict(cell, intervention=arm)
                for i in range(config.n_runs):
                    trials.append(
                        Trial(
                            id=f"{cell['model']}_{cell['item']}_{cell['frame']}_{'fix' if arm != 'none' else 'none'}_r{i}",
                            cell=full_cell,
                            messages=[{"role": "user", "content": prompt}],
                            system=system,
                            model_id=cell["model"],
                            temperature=config.temperature,
                            parse=make_parse(cell["item"]),
                            run_index=i,
                        )
                    )
        return trials

    def leaderboard_rows(
        self, config: ExperimentConfig, results: List[TrialResult]
    ) -> List[Dict[str, Any]]:
        """TriageGap per model = mean absolute framing effect across the
        equivalence items (the control item is excluded — it isn't equivalent).
        No-intervention arm only. Probe id = 'framing'."""
        from triagebench.models import REGISTRY

        baseline = [r for r in results if r.cell.get("intervention", "none") == "none"]
        grouped = metrics.group_by_cell(baseline, axes=["model", "item", "frame"])
        rates: Dict[tuple, float] = {}
        ns: Dict[tuple, int] = {}
        for key, group in grouped.items():
            d = dict(key)
            p, n = metrics.proportion(group, "chose_target", True)
            rates[(d["model"], d["item"], d["frame"])] = p
            ns[(d["model"], d["item"])] = ns.get((d["model"], d["item"]), 0) + n
        models_seen = sorted({k[0] for k in rates})
        items = [i for i in ITEMS if not i.startswith("control")]
        rows = []
        for model in models_seen:
            effects, total_n = [], 0
            for item in items:
                g = rates.get((model, item, "gain"))
                l = rates.get((model, item, "loss"))
                if g is None or l is None:
                    continue
                effects.append(abs(g - l))
                total_n += ns.get((model, item), 0)
            if not effects:
                continue
            gap = sum(effects) / len(effects)
            per_level_n = max(1, total_n // (2 * len(effects)))
            lo, hi = metrics.wilson_ci(gap, per_level_n)
            rows.append({
                "model": REGISTRY[model].api_model if model in REGISTRY else model,
                "probe": "framing",
                "domain": "general",
                "axis": "frame",
                "ceteris_gap": round(gap, 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "n_per_level": per_level_n,
            })
        return rows

    def report(
        self, config: ExperimentConfig, results: List[TrialResult], out_dir: str
    ) -> Dict[str, Any]:
        out = Path(out_dir)
        effects = metrics.get_metric("framing_effect")(results)
        _write_report(out / "report.md", config, effects)
        _plot_effects(effects, out)
        return {
            "experiment": self.name,
            "models": config.models,
            "n_results": len(results),
            "n_errors": sum(1 for r in results if r.error),
            "framing_effects": effects,
        }


@metrics.metric("framing_effect")
def framing_effect(results: List[TrialResult]) -> List[Dict[str, Any]]:
    """Per (model, item, intervention): target-choice rate under each frame,
    the gain-loss difference, and the favorability shift."""
    grouped = metrics.group_by_cell(
        results, axes=["model", "item", "intervention", "frame"]
    )
    cells: Dict[tuple, Dict[str, Any]] = defaultdict(dict)
    for key, group in grouped.items():
        d = dict(key)
        p, n = metrics.proportion(group, "chose_target", True)
        cells[(d["model"], d["item"], d["intervention"])][d["frame"]] = {
            "rate": p,
            "n": n,
            "fav": metrics.mean(group, "favorability"),
        }

    rows = []
    for (model, item, arm), frames in sorted(cells.items()):
        if "gain" not in frames or "loss" not in frames:
            continue
        g, l = frames["gain"], frames["loss"]
        rows.append({
            "model": model,
            "item": item,
            "intervention": "none" if arm == "none" else "equivalence_note",
            "target_rate_gain": round(g["rate"], 4),
            "target_rate_loss": round(l["rate"], 4),
            "framing_effect": round(g["rate"] - l["rate"], 4),
            "favorability_shift": round(g["fav"] - l["fav"], 2),
            "n_per_frame": g["n"],
        })
    return rows


def _write_report(path: Path, config, effects):
    lines = [
        "# Framing report",
        "",
        f"Models: {', '.join(f'`{m}`' for m in config.models)} | "
        f"n_runs/cell: {config.n_runs} | temperature: {config.temperature}",
        "",
        "Framing effect = P(target | gain frame) - P(target | loss frame).",
        "Zero = frame-invariant. `control_nonequivalent` is NOT equivalent;",
        "a nonzero difference there is correct behavior.",
        "",
        "| Model | Item | Arm | Gain | Loss | Effect | Fav shift |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in effects:
        lines.append(
            f"| {r['model']} | {r['item']} | {r['intervention']} "
            f"| {r['target_rate_gain']:.0%} | {r['target_rate_loss']:.0%} "
            f"| {r['framing_effect']:+.0%} | {r['favorability_shift']:+.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _plot_effects(effects, out: Path):
    """One grouped chart per arm: framing effect by item, one series per model."""
    arms = sorted({r["intervention"] for r in effects})
    for arm in arms:
        sub = [r for r in effects if r["intervention"] == arm]
        items = sorted({r["item"] for r in sub})
        series = {}
        for model in sorted({r["model"] for r in sub}):
            vals = {r["item"]: r["framing_effect"] for r in sub if r["model"] == model}
            series[model] = (list(range(len(items))), [vals.get(i, 0) for i in items])
        fig = plot.lines(
            series,
            title=f"Framing effect by item ({arm})",
            xlabel="item",
            ylabel="P(target|gain) - P(target|loss)",
            hlines={"frame-invariant": 0.0},
        )
        ax = fig.axes[0]
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(items, rotation=20, ha="right", fontsize=8)
        plot.save(fig, str(out / f"framing_effect_{arm}.png"))


experiment = FramingExperiment()
