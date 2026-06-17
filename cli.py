"""TriageBench command-line entry point.

    triagebench run        configs/triage_gender_age.yaml
    triagebench report     configs/triage_gender_age.yaml [JSONL]
    triagebench leaderboard configs/*.yaml -o leaderboard.json
    triagebench models

`report` rebuilds tables/charts from an existing JSONL with no API calls.
`leaderboard` reduces every probe to a single comparable TriageGap and emits the
JSON the public dashboard renders. `models` lists the registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from triagebench import load_config
from triagebench.experiment import load_experiment
from triagebench.models import REGISTRY
from triagebench.runner import Runner, latest_jsonl, load_results


def _out_dir(config_path: str, experiment: str) -> Path:
    d = Path("results") / experiment / "reports" / Path(config_path).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_run(args) -> int:
    config = load_config(args.config)
    exp = load_experiment(config.experiment)
    trials = exp.build_trials(config)
    print(
        f"Probe: {config.experiment} | models={config.models} | "
        f"cells={len(config.cells())} | n_runs={config.n_runs} | trials={len(trials)}"
    )
    runner = Runner(
        experiment=config.experiment,
        concurrency=config.concurrency,
        use_cache=config.use_cache,
        label=Path(args.config).stem,
    )
    results, _ = runner.run(trials)
    out = _out_dir(args.config, config.experiment)
    summary = exp.report(config, results, str(out))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nReport written to {out}")
    return 0


def cmd_report(args) -> int:
    config = load_config(args.config)
    exp = load_experiment(config.experiment)
    jsonl = args.jsonl or latest_jsonl(
        "results", config.experiment, label=Path(args.config).stem
    )
    if not jsonl:
        print(f"No JSONL found for {config.experiment}. Run it first.", file=sys.stderr)
        return 1
    print(f"Reporting from {jsonl}")
    results = load_results(str(jsonl))
    out = _out_dir(args.config, config.experiment)
    summary = exp.report(config, results, str(out))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Report written to {out}")
    return 0


def cmd_leaderboard(args) -> int:
    """Reduce every named probe config to comparable TriageGap rows."""
    rows = []
    for cfg_path in args.configs:
        config = load_config(cfg_path)
        exp = load_experiment(config.experiment)
        jsonl = latest_jsonl("results", config.experiment, label=Path(cfg_path).stem)
        if not jsonl:
            print(f"  skip {cfg_path}: no results yet", file=sys.stderr)
            continue
        results = load_results(str(jsonl))
        get_rows = getattr(exp, "leaderboard_rows", None)
        if get_rows is None:
            print(f"  skip {cfg_path}: probe has no leaderboard_rows()", file=sys.stderr)
            continue
        new = get_rows(config, results)
        for r in new:
            r.setdefault("config", Path(cfg_path).stem)
        rows.extend(new)
        print(f"  {cfg_path}: {len(new)} rows")

    board = {
        "benchmark": "TriageBench",
        "snapshot_date": date.today().isoformat(),
        "metric": "TriageGap",
        "metric_definition": "max-min spread of the decision rate across the swapped axis; 0 = invariant",
        "rows": sorted(rows, key=lambda r: (r["probe"], r["model"])),
    }
    out = Path(args.output)
    out.write_text(json.dumps(board, indent=2, default=str))

    # Also emit a human-readable markdown table (lower TriageGap = more consistent).
    md_path = out.with_suffix(".md")
    md_path.write_text(_leaderboard_markdown(board))

    if not rows:
        print("\nNo results found for any probe. Run them first: ./scripts/leaderboard.sh",
              file=sys.stderr)
    print(f"\nLeaderboard: {len(rows)} rows -> {out} and {md_path}")
    return 0


_PROBE_LABELS = {
    "triage_neuro_sex": "Gender (man vs woman)",
    "triage_neuro_language": "Language (English vs Japanese)",
    "triage_neuro_zipcode": "SES (rich vs poor ZIP)",
}


def _leaderboard_markdown(board: dict) -> str:
    """Render the board as a probe (row) x model (column) matrix.
    Each cell is the TriageGap in points; bold = statistically significant
    (two-proportion z-test, p<0.05). Lower = more consistent."""
    rows = board["rows"]
    if not rows:
        return (
            "# TriageBench leaderboard\n\n"
            "No probe results found yet. Run the probes first, then rebuild the board:\n\n"
            "```bash\n./scripts/leaderboard.sh\n```\n"
        )
    probes = sorted({r["probe"] for r in rows})
    models = sorted({r["model"] for r in rows})
    cell = {(r["probe"], r["model"]): r for r in rows}

    header = "| Probe \\ Model | " + " | ".join(models) + " |"
    sep = "|---|" + "|".join(["---:"] * len(models)) + "|"
    lines = [
        "# TriageBench leaderboard",
        "",
        f"*Snapshot {board['snapshot_date']}. Cell = **TriageGap** in points "
        "(gap in ER-referral rate across the swapped attribute). "
        "**Lower = more consistent.** **Bold** = significant at p<0.05 "
        "(two-proportion z-test). Each cell shows gap (p-value).*",
        "",
        header,
        sep,
    ]
    for probe in probes:
        label = _PROBE_LABELS.get(probe, probe)
        cells = []
        for m in models:
            r = cell.get((probe, m))
            if not r:
                cells.append("—")
                continue
            gap = round(r.get("ceteris_gap", 0) * 100, 1)
            pv = r.get("p_value")
            pstr = f"p={pv:.3f}" if isinstance(pv, (int, float)) else "p=n/a"
            txt = f"{gap} ({pstr})"
            if isinstance(pv, (int, float)) and pv < 0.05:
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"*n={rows[0].get('n_per_level','?')} per condition. "
                 "TriageGap = max-min ER-rate spread across the attribute's levels.*")
    return "\n".join(lines)


def cmd_models(args) -> int:
    for alias, spec in sorted(REGISTRY.items()):
        print(f"{alias:<16} {spec.provider:<10} {spec.api_model}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="triagebench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a probe from a config")
    pr.add_argument("config")
    pr.set_defaults(fn=cmd_run)

    prep = sub.add_parser("report", help="rebuild tables/charts from cached JSONL")
    prep.add_argument("config")
    prep.add_argument("jsonl", nargs="?", default=None)
    prep.set_defaults(fn=cmd_report)

    plb = sub.add_parser("leaderboard", help="emit the public TriageGap leaderboard JSON")
    plb.add_argument("configs", nargs="+")
    plb.add_argument("-o", "--output", default="leaderboard.json")
    plb.set_defaults(fn=cmd_leaderboard)

    pm = sub.add_parser("models", help="list the model registry")
    pm.set_defaults(fn=cmd_models)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
