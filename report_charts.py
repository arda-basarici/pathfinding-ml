"""report_charts.py — print-quality figures for the PDF report.

Each function takes the gathered report data (see report_data.gather) and an output
path, renders one figure at print resolution using a shared visual identity, and returns
the path. Crimson = the learned heuristic (the thing under study); blue = the Manhattan
baseline (the incumbent).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#C0392B"      # crimson — the learned heuristic
COUNTER = "#2980B9"     # blue — the Manhattan baseline
INK = "#2B2B2B"
GREY_DARK = "#7F8C8D"
GREY = "#BDC3C7"
GREY_LIGHT = "#ECF0F1"
DPI = 200


def _base():
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelcolor": INK, "axes.edgecolor": GREY, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GREY_LIGHT, "grid.linewidth": 0.9,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    })


def _finish(fig, ax, title, subtitle):
    ax.set_title(title + "\n", loc="left", fontsize=13, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=GREY_DARK, va="bottom")
    fig.tight_layout()


# --- small extractors over a record -----------------------------------------
def _nodes(s, combo): return s[combo]["mean_nodes_expanded"]
def _gap(s, combo): return s[combo]["mean_optimality_gap"] * 100
def _win(rec):
    s = rec["axis2_summary"]
    return (_nodes(s, "astar+learned") / _nodes(s, "astar+manhattan") - 1) * 100
def _learned_gap(rec): return _gap(rec["axis2_summary"], "astar+learned")
def _mae(rec): return rec["axis1_quality"]["learned"]["mae"]
def _overest(rec): return rec["axis1_quality"]["learned"]["frac_overestimated"] * 100


# --- figures ----------------------------------------------------------------
def fig_within_group(data, path):
    _base()
    rec = data["baseline"]
    groups = ["Pooled", "Scattered", "Structured"]
    sources = [rec["axis2_summary"], rec["by_style"]["scattered"], rec["by_style"]["structured"]]
    learned = [_nodes(s, "astar+learned") for s in sources]
    manh = [_nodes(s, "astar+manhattan") for s in sources]
    gaps = [_gap(s, "astar+learned") for s in sources]

    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    b1 = ax.bar(x - w / 2, learned, w, label="A* + learned", color=ACCENT)
    b2 = ax.bar(x + w / 2, manh, w, label="A* + Manhattan", color=COUNTER)
    ax.bar_label(b1, fmt="%.0f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.0f", fontsize=8, padding=2)
    for xi, lv, g in zip(x, learned, gaps):
        ax.text(xi - w / 2, lv + max(learned) * 0.06, f"gap {g:.1f}%", ha="center",
                fontsize=8, color=ACCENT, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Mean nodes expanded (lower = faster)")
    ax.set_ylim(0, max(manh) * 1.25)
    ax.legend(fontsize=9, loc="upper left")
    _finish(fig, ax, "The average hides two opposite stories",
            "Six-feature model. Pooled looks like a wash; split by maze type it reverses")
    fig.savefig(path); plt.close(fig); return path


def fig_global_fix(data, path):
    _base()
    base, glob = data["baseline"], data["global"]
    labels = ["6 features", "+ global density"]
    wins = [_win(base), _win(glob)]
    gaps = [_learned_gap(base), _learned_gap(glob)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    b = ax1.bar(labels, wins, color=[GREY, ACCENT], width=0.6)
    ax1.bar_label(b, fmt="%+.1f%%", fontsize=10, padding=3)
    ax1.set_title("Nodes vs Manhattan (lower = faster)", loc="left", fontsize=11)
    ax1.set_ylabel("% change in nodes expanded")
    ax1.axhline(0, color=INK, linewidth=0.8)

    b2 = ax2.bar(labels, gaps, color=[GREY, ACCENT], width=0.6)
    ax2.bar_label(b2, fmt="%.2f%%", fontsize=10, padding=3)
    ax2.set_title("Optimality gap (lower = better paths)", loc="left", fontsize=11)
    ax2.set_ylabel("mean optimality gap (%)")
    ax2.set_ylim(0, max(gaps) * 1.3 + 0.5)
    fig.suptitle("One feature turns a wash into a win", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path); plt.close(fig); return path


def _importance_panel(ax, rec, title, top=6):
    items = rec["feature_importance"][:top][::-1]
    names = [d["feature"].replace("_", " ") for d in items]
    vals = [d["importance_mean"] for d in items]
    colors = [ACCENT if "global" in n else GREY_DARK for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("permutation importance (MAE units)")


def fig_importance(data, path):
    _base()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4))
    _importance_panel(ax1, data["baseline"], "Six features")
    _importance_panel(ax2, data["global"], "+ global density")
    fig.suptitle("What the model leans on — and what global density makes redundant",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path); plt.close(fig); return path


def fig_mix_sweep(data, path):
    _base()
    sfs = sorted(data["mix"].keys())
    wins = [_win(data["mix"][sf]) for sf in sfs]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot([s * 100 for s in sfs], wins, "-o", color=ACCENT, linewidth=2, markersize=7)
    for s, wv in zip(sfs, wins):
        ax.annotate(f"{wv:+.0f}%", (s * 100, wv), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=INK)
    ax.set_xlabel("% structured (corridor) mazes in the set")
    ax.set_ylabel("A* learned vs Manhattan, nodes (%)")
    ax.axhline(0, color=INK, linewidth=0.8)
    _finish(fig, ax, "The headline number just tracks the maze mix",
            "+global model. The single pooled figure is an artifact of the 50/50 default")
    fig.savefig(path); plt.close(fig); return path


def fig_transfer(data, path):
    _base()
    s2t, t2s = data["transfer_s2t"], data["transfer_t2s"]
    labels = ["scattered → structured", "structured → scattered"]
    wins = [_win(s2t), _win(t2s)]
    gaps = [_learned_gap(s2t), _learned_gap(t2s)]
    over = [_overest(s2t), _overest(t2s)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.3))
    b = ax1.bar(labels, wins, color=[COUNTER, ACCENT], width=0.6)
    ax1.bar_label(b, fmt="%+.0f%%", fontsize=10, padding=3)
    ax1.axhline(0, color=INK, linewidth=0.8)
    ax1.set_title("Nodes vs Manhattan", loc="left", fontsize=11)
    ax1.set_ylabel("% change in nodes")
    ax1.tick_params(axis="x", labelsize=8)

    b2 = ax2.bar(labels, gaps, color=[COUNTER, ACCENT], width=0.6)
    ax2.bar_label(b2, fmt="%.1f%%", fontsize=10, padding=3)
    ax2.set_title("Optimality gap", loc="left", fontsize=11)
    ax2.set_ylabel("mean optimality gap (%)")
    ax2.tick_params(axis="x", labelsize=8)
    for xi, ov in zip(range(2), over):
        ax2.text(xi, max(gaps) * 0.5, f"overest\n{ov:.0f}%", ha="center", fontsize=8,
                 color=GREY_DARK)
    fig.suptitle("Off its training distribution, the model fails — in opposite ways",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path); plt.close(fig); return path


def fig_seed_robustness(data, path):
    _base()
    bw = [_win(r) for r in data["baseline_seeds"]]
    gw = [_win(r) for r in data["global_seeds"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter([0] * len(bw), bw, s=70, color=GREY_DARK, zorder=3)
    ax.scatter([1] * len(gw), gw, s=70, color=ACCENT, zorder=3)
    ax.plot([-0.18, 0.18], [np.mean(bw)] * 2, color=GREY_DARK, lw=2)
    ax.plot([0.82, 1.18], [np.mean(gw)] * 2, color=ACCENT, lw=2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["6 features", "+ global density"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("A* learned vs Manhattan, nodes (%)")
    _finish(fig, ax, "The win holds across seeds",
            "Each dot is one random seed; bar is the mean. Tight clusters = a real effect, not noise")
    fig.savefig(path); plt.close(fig); return path


def fig_minimal(data, path):
    _base()
    keys = [("min2", 2), ("min3", 3), ("global", 7)]
    # MAE goes into the x-tick label (below the axis) so it never collides with the title.
    labels = [f"{count} features\nMAE {_mae(data[key]):.1f}" for key, count in keys]
    wins = [_win(data[key]) for key, _ in keys]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b = ax.bar(labels, wins, color=[ACCENT, ACCENT, GREY], width=0.6)
    ax.bar_label(b, fmt="%+.1f%%", fontsize=10, padding=3)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("A* learned vs Manhattan, nodes (%)")
    ax.set_ylim(min(wins) * 1.30, 1.0)            # headroom below for the labels; top near 0
    _finish(fig, ax, "Three features do the work of seven",
            "{manhattan, euclidean, global density} matches the full set; the rest is redundant")
    fig.savefig(path); plt.close(fig); return path
