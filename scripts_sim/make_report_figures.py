"""Figures for docs/report_sim/librarian_sim.tex (Times-style fonts to match the report).

    python scripts_sim/make_report_figures.py      # needs results/sim_runs/librarian/log.csv (run_librarian.py),
                                                   # results/ablation/results.json (ablation.py) and the demo video
Writes docs/report_sim/fig_storyboard.jpg, fig_run.pdf, fig_ablation.pdf.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from librarian_media import lowpass  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report_sim"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
COL = {"left": "#1f77b4", "right": "#d62728"}


def load_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def phase_starts(L):
    starts, last = [], None
    for t, p in zip(L["t"], L["phase"]):
        if p != last:
            starts.append((float(t), p))
            last = p
    return starts


def shade(ax, starts, t_end, labels=True):
    keys = [("approach", "A"), ("insert", "I"), ("controller: close", "C"), ("controller: pull", "P"),
            ("controller: lift", "L"), ("controller: hold", "H"), ("carry", "T"), ("shelve", "S"), ("retreat", "R")]
    for k, (t0, p) in enumerate(starts):
        t1 = starts[k + 1][0] if k + 1 < len(starts) else t_end
        tag = next((s for key, s in keys if p.startswith(key)), None)
        if tag is None:
            continue
        if k % 2 == 0:
            ax.axvspan(t0, t1, color="#000000", alpha=0.045, lw=0)
        if labels:
            ax.text((t0 + t1) / 2, 1.02, tag, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)


def fig_run(L, path):
    t = L["t"]
    starts = phase_starts(L)
    fig, ax = plt.subplots(4, 1, figsize=(3.45, 4.6), sharex=True, gridspec_kw={"hspace": 0.18})
    for side in ("left", "right"):
        ax[0].plot(t, L[f"{side}_normal"], color=COL[side], label=f"$N$ {side}")
        ax[0].plot(t, L[f"{side}_shear"], color=COL[side], ls="--", label=f"$|T|$ {side}")
        ax[1].plot(t, np.clip(L[f"{side}_margin"], -0.2, 1.0), color=COL[side], label=f"$q$ {side}")
        ax[2].plot(t, L[f"{side}_comp_um"], color=COL[side], label=f"compression {side}")
        ax[2].plot(t, L[f"{side}_sdisp_um"], color=COL[side], ls="--", label=f"shear disp. {side}")
    ax[1].axhline(0.5, color="0.5", lw=0.6, ls=":")
    ax1b = ax[1].twinx()
    ax1b.plot(t, 10 + 115 * L["gripper_force"] / 255, color="#e0a000", lw=1.0, label="grip setting (N/finger)")
    ax1b.set_ylabel("grip set (N)", color="#a07000")
    ax1b.tick_params(axis="y", colors="#a07000", labelsize=6.5)
    ax[3].plot(t, -lowpass(L["fz"], t), color="#2ca02c", label="$-F_z$ arm F/T, 3 Hz (N)")
    ax[3].plot(t, L["slip_mm"], color="#ff7f0e", label="slip (mm)")
    ax[3].plot(t, L["slip_deg"], color="#9467bd", label="rotation (deg)")
    for a, lab in zip(ax, ["force (N)", "margin $q$", "PDMS (µm)", "N, mm, deg"]):
        a.set_ylabel(lab)
        a.grid(alpha=0.25, lw=0.4)
        shade(a, starts, t[-1], labels=a is ax[0])
    for a in (ax[0], ax[2], ax[3]):
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.55 * (hi - lo))  # headroom for the legend
    ax[0].legend(ncol=4, loc="upper left", frameon=False, columnspacing=0.8, handlelength=1.5)
    ax[1].legend(loc="lower left", frameon=False, ncol=2)
    ax1b.legend(loc="lower right", frameon=False)
    ax[2].legend(ncol=2, loc="upper left", frameon=False, columnspacing=0.8, handlelength=1.5)
    ax[3].legend(ncol=3, loc="upper left", frameon=False, columnspacing=0.6, handlelength=1.5)
    ax[1].set_ylim(-0.25, 1.05)
    ax[-1].set_xlabel("time (s)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(results, path):
    ctrls = ["skin", "fixed-10N", "fixed-40N", "fixed-70N", "fixed-125N"]
    colors = {"skin": "#d62728", "fixed-10N": "#9ecae1", "fixed-40N": "#6baed6", "fixed-70N": "#2171b5",
              "fixed-125N": "#08306b"}
    marker = {"shelved": "o", "put back": "s", "dropped": "x", "misplaced": "^"}
    mus = sorted({r["mu"] for r in results}, reverse=True)
    fig, ax = plt.subplots(2, len(mus), figsize=(3.45, 3.3), sharex=True, sharey="row",
                           gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    ax = np.atleast_2d(ax)
    for k, mu in enumerate(mus):
        for c in ctrls:
            rs = sorted((r for r in results if r["controller"] == c and r["mu"] == mu), key=lambda r: r["mass"])
            if not rs:
                continue
            off = 0.025 * (ctrls.index(c) - 2)
            x = np.array([r["mass"] for r in rs]) + off
            ax[0, k].plot(x, [r["squeeze_hold"] for r in rs], color=colors[c], lw=0.9, label=c)
            ax[1, k].plot(x, [max(r["slip_mm"], 0.05) for r in rs], color=colors[c], lw=0.9)
            for xi, r in zip(x, rs):
                filled = r["outcome"] == "shelved"
                for row, val in ((0, r["squeeze_hold"]), (1, max(r["slip_mm"], 0.05))):
                    ax[row, k].plot(xi, val, marker[r["outcome"]], color=colors[c], ms=3.5,
                                    mfc=colors[c] if filled else "white", mew=0.8)
        ax[0, k].set_title(rf"true $\mu$ = {mu:g}", pad=2)
        ax[1, k].set_yscale("log")
        ax[1, k].set_xlabel("book mass (kg)")
        for a in ax[:, k]:
            a.grid(alpha=0.25, lw=0.4)
    ax[0, 0].set_ylabel("squeeze holding (N/blade)")
    ax[1, 0].set_ylabel("slip in grasp (mm)")
    h = [plt.Line2D([], [], color=colors[c], lw=1.4, label=c) for c in ctrls]
    seen = {r["outcome"] for r in results}
    h += [plt.Line2D([], [], color="0.2", marker=mk, ls="none", ms=3.5, mfc="0.2" if o == "shelved" else "white",
                     label=o) for o, mk in marker.items() if o in seen]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=6,
               columnspacing=0.8, handlelength=1.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def storyboard(video, L, path, crop=(560, 54, 1360, 654)):
    """Six frames of the demo video (main MuJoCo view), at the start of the main phases."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    starts = phase_starts(L)
    pick = [("approach", 1.0, "approach"), ("insert", 1.6, "insert beside the book"),
            ("controller: pull", 2.0, "squeeze and pull out"), ("carry", 2.6, "carry to shelf B"),
            ("shelve", 4.5, "slide in, set down"), ("retreat", 1.5, "release and retreat")]
    frames = []
    for key, dt, label in pick:
        t0 = next((t for t, p in starts if p.startswith(key)), None)
        if t0 is None:
            continue
        tmp = OUT / "_frame.png"
        subprocess.run([exe, "-y", "-loglevel", "error", "-ss", f"{t0 + dt:.2f}", "-i", str(video), "-frames:v", "1",
                        str(tmp)], check=True)
        frames.append((Image.open(tmp).convert("RGB").crop(crop).resize((400, 300), Image.LANCZOS), label))
        tmp.unlink()
    W, H = 3 * 400, 2 * 300
    sheet = Image.new("RGB", (W, H), "white")
    try:
        font = ImageFont.truetype("NimbusRoman-Bold.otf", 22)
    except OSError:
        font = ImageFont.load_default()
    for k, (im, label) in enumerate(frames):
        x, y = (k % 3) * 400, (k // 3) * 300
        sheet.paste(im, (x, y))
        d = ImageDraw.Draw(sheet)
        d.rectangle((x + 6, y + 6, x + 12 + 12 * len(label) + 30, y + 34), fill=(255, 255, 255))
        d.text((x + 12, y + 8), f"({'abcdef'[k]}) {label}", font=font, fill=(0, 0, 0))
    sheet.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", type=Path, default=ROOT / "results" / "sim_runs" / "librarian" / "log.csv")
    ap.add_argument("--ablation", type=Path, default=ROOT / "results" / "ablation" / "results.json")
    ap.add_argument("--video", type=Path, default=ROOT / "results" / "librarian.mp4")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    L = load_log(args.log)
    fig_run(L, OUT / "fig_run.pdf")
    if args.ablation.exists():
        fig_ablation(json.loads(args.ablation.read_text()), OUT / "fig_ablation.pdf")
    if args.video.exists():
        storyboard(args.video, L, OUT / "fig_storyboard.jpg")
    print(f"figures -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
