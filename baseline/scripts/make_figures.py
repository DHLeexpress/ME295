"""Figures for the VLA baseline report from baseline/results/summary.json (run report.py first).

    python baseline/scripts/make_figures.py      # -> baseline/results/fig_success.{png,pdf}, fig_funnel.{png,pdf}

Colours: the reference categorical palette, slots 1-3 (validated: CVD dE >= 9.2, normal-vision dE >= 27.6; slot 3 is
below 3:1 contrast on the surface, so every bar carries a visible value label and RESULTS_TABLES.md is the table view).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "baseline" / "results"
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POLICIES = [  # fixed slot per entity, never re-ordered by rank
    ("pi05_ft", "π0.5 fine-tuned (LoRA, 251 demos)", "#2a78d6"),
    ("pi05_droid", "π0.5-DROID zero-shot", "#eb6834"),
    ("expert", "Scripted expert (privileged)", "#1baf7a"),
]
SUITES = [("nominal", "Nominal\n(simulator scene)"), ("random", "Randomized\nlayouts"),
          ("narrow", "Narrow gaps\n6.5-7.5 mm"), ("heavy_lowmu", "Heavy 1.6-2.4 kg\nPDMS μ 0.5")]
STAGES = [("approached", "approached"), ("contact", "pads on book"), ("moved", "pulled > 2 cm"),
          ("extracted", "out of shelf"), ("lifted", "lifted 3 cm"), ("rate", "success")]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    summary = json.loads((RES / "summary.json").read_text())
    def step_of(name):
        return int(name.rsplit("_step", 1)[1]) if "_step" in name else -1
    pick, curve = {}, {}
    for key, _, _ in POLICIES:  # the policy folder may carry a checkpoint tag, e.g. pi05_ft_step9999
        names = sorted((n for n in summary if n == key or n.startswith(key + "_step")), key=step_of)
        if names:
            pick[key] = summary[names[-1]]
        if len(names) > 1:
            curve[key] = [(step_of(n), summary[n]) for n in names if step_of(n) >= 0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": TEXT2})

    # ---- success rate per suite
    fig, ax = plt.subplots(figsize=(9.0, 3.6), facecolor=SURFACE)
    style(ax)
    w = 0.26
    for j, (key, label, color) in enumerate(POLICIES):
        for i, (suite, _) in enumerate(SUITES):
            x = i + (j - 1) * w
            r = pick.get(key, {}).get(suite)
            if r is None:
                ax.text(x, 0.02, "not run", rotation=90, ha="center", va="bottom", fontsize=7, color=TEXT2)
                continue
            ax.bar(x, max(r["rate"], 0.008), w, color=color, edgecolor=SURFACE, linewidth=2,   # 0 %: a visible stub
                   label=label if i == 0 else None)
            lo, hi = r["ci95"]
            ax.plot([x, x], [lo, hi], color=TEXT2, lw=1.0)
            ax.text(x, hi + 0.025, f"{100 * r['rate']:.0f}%", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    ax.set_xticks(range(len(SUITES)), [s[1] for s in SUITES], color=TEXT)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("task success (95% CI)")
    ax.set_title("Pick up the designated book from the packed shelf (UR10e + blade/PDMS gripper, MuJoCo)",
                 fontsize=10, color=TEXT, loc="left", pad=26)
    leg = ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0),
                    borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"fig_success.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ---- stage funnels where the policies differ (the nominal one is degenerate: 100 / 100 / 0 %)
    panels = [("random", "Randomized layouts"), ("heavy_lowmu", "Heavy 1.6-2.4 kg, PDMS μ 0.5")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.3), facecolor=SURFACE, sharey=True)
    xs = np.arange(len(STAGES))
    widths = {"expert": 6, "pi05_ft": 2, "pi05_droid": 2}       # expert drawn as a wide band underneath
    for ax, (suite, title) in zip(axes, panels):
        style(ax)
        for key, label, color in sorted(POLICIES, key=lambda p: p[0] != "expert"):
            r = pick.get(key, {}).get(suite)
            if r is None:
                continue
            ys = [r[k] for k, _ in STAGES]
            ax.plot(xs, ys, color=color, lw=widths[key], alpha=0.45 if key == "expert" else 1.0,
                    marker="o", ms=4 if key == "expert" else 7, mec=SURFACE, mew=1.5, label=label,
                    solid_capstyle="round", zorder=1 if key == "expert" else 3)
            ax.text(xs[-1] + 0.15, ys[-1], f"{100 * ys[-1]:.0f}%", va="center", fontsize=8, color=TEXT)
        ax.set_xticks(xs, [st[1] for st in STAGES], color=TEXT, fontsize=8, rotation=20, ha="right")
        ax.set_xlim(-0.3, len(STAGES) - 0.4)
        ax.set_ylim(-0.04, 1.08)
        ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
        ax.set_title(title, fontsize=9.5, color=TEXT, loc="left")
    axes[0].set_ylabel("episodes reaching the stage")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [lab for _, lab, _ in POLICIES if lab in labels]          # same order as the success figure
    handles = [handles[labels.index(lab)] for lab in order]
    labels = order
    leg = fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"fig_funnel.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ---- success vs fine-tuning steps (only if several checkpoints were evaluated)
    if "pi05_ft" in curve:
        fig, ax = plt.subplots(figsize=(6.0, 3.0), facecolor=SURFACE)
        style(ax)
        for (suite, slabel), mk in zip(SUITES, ("o", "s", "^", "D")):
            pts = [(st, r[suite]) for st, r in curve["pi05_ft"] if suite in r]
            if len(pts) < 2:
                continue
            xs_, ys_ = [p[0] for p in pts], [p[1]["rate"] for p in pts]
            ax.plot(xs_, ys_, color="#2a78d6", lw=2, marker=mk, ms=7, mec=SURFACE, mew=2,
                    alpha=1.0 if suite == "nominal" else 0.55, label=slabel.replace("\n", " "))
        ax.set_xlabel("LoRA fine-tuning steps (batch 32)")
        ax.set_ylabel("task success")
        ax.set_ylim(-0.04, 1.08)
        ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
        ax.set_title("π0.5 fine-tuned: success vs training steps", fontsize=10, color=TEXT, loc="left")
        leg = ax.legend(frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(TEXT)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(RES / f"fig_steps.{ext}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
    print("figures ->", RES)


if __name__ == "__main__":
    main()
