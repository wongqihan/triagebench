"""Standard chart styles (matplotlib).

Thin helpers so every experiment's charts share a look. Each returns the Figure
so experiments can tweak before saving.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless; experiments save to PNG
import matplotlib.pyplot as plt

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def bars(
    labels: Sequence[str],
    values: Sequence[float],
    errs: Optional[Sequence[Tuple[float, float]]] = None,
    title: str = "",
    ylabel: str = "",
    ylim: Optional[Tuple[float, float]] = None,
):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4.5))
    x = range(len(labels))
    yerr = None
    if errs:
        # Clamp at zero: CI bounds can dip past the point value by float epsilon
        # when the proportion is exactly 0 or 1.
        lo = [max(0.0, v - e[0]) for v, e in zip(values, errs)]
        hi = [max(0.0, e[1] - v) for v, e in zip(values, errs)]
        yerr = [lo, hi]
    ax.bar(x, values, color=PALETTE[0], yerr=yerr, capsize=4, alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax)
    fig.tight_layout()
    return fig


def lines(
    series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    hlines: Optional[Dict[str, float]] = None,
):
    """series: name -> (x, y). hlines: label -> y for reference lines."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, (xs, ys)) in enumerate(series.items()):
        ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=PALETTE[i % len(PALETTE)], label=name)
    if hlines:
        for label, y in hlines.items():
            ax.axhline(y, ls="--", lw=1.2, color="#6b7280", alpha=0.8)
            ax.text(0.0, y, f" {label}", va="bottom", ha="left", fontsize=8, color="#6b7280")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def facet_lines(
    facets: Dict[str, Dict[str, Tuple[Sequence[float], Sequence[float]]]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    hlines: Optional[Dict[str, float]] = None,
):
    """One subplot per facet key; each facet is a {series_name: (x, y)} dict."""
    n = len(facets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), sharey=True, squeeze=False)
    for ax, (fname, series) in zip(axes[0], facets.items()):
        for i, (name, (xs, ys)) in enumerate(series.items()):
            ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=PALETTE[i % len(PALETTE)], label=name)
        if hlines:
            for _, y in hlines.items():
                ax.axhline(y, ls="--", lw=1.0, color="#6b7280", alpha=0.7)
        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel)
        _style(ax)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def save(fig, path: str, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
