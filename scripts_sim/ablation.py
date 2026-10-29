"""Ablation: when does the skin help? The librarian task (shelf A -> shelf B) with and without the skin in the loop,
across book mass and the true PDMS-cover friction.

    python scripts_sim/ablation.py                  # full grid in parallel -> results/ablation/, docs/figures/sim_ablation.png
    python scripts_sim/ablation.py --quick          # 6 runs, for a smoke test

Controllers
  skin      ShearMarginGrasp: starts at the lowest force (10 N per finger) and raises the Robotiq force register
            while the skin's patch margin q < 0.5, pauses the arm while q < 0.2, aborts (puts the book back) if
            q < 0.2 persists at full force. Its margin assumes mu = 1.0, whatever the true friction is.
  fixed-F   FixedForceGrasp: the same motions with a constant force register (no skin), F per finger.

Per run: outcome (shelved / put back / dropped or misplaced), the book's slip and rotation in the grasp from the
grasp until the set-down in shelf B, the squeeze on each blade (sum of the cells' contact normal forces; peak and
median while holding), time, and how far other books moved.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_sim"))

import numpy as np  # noqa: E402

REGISTER = {"fixed-10N": 0, "fixed-40N": 67, "fixed-70N": 133, "fixed-125N": 255}  # 10 + 115 * reg / 255 N
CONTROLLERS = ["skin", *REGISTER]
MASSES = [0.4, 0.8, 1.6, 2.4]
FRICTIONS = [1.0, 0.5]
MU_ASSUMED = 1.0


def one(cfg: dict) -> dict:
    import run_librarian as rl
    ctrl, mass, mu = cfg["controller"], cfg["mass"], cfg["mu"]
    rob, gripper, skins, rec, controller = rl.make_sim(
        "margin" if ctrl == "skin" else "fixed", grip_force=REGISTER.get(ctrl, 0), mu=mu, mu_assumed=MU_ASSUMED,
        book_mass=mass, verbose=False)
    rob.sleep(0.5)
    try:
        done = rl.task(rob, gripper, controller, rob.m, rec=rec, skins=skins, say=lambda s: rob._say(s))
        rob.sleep(1.0)
        error = ""
    except Exception as e:  # e.g. an IK failure: count it, keep the grid going
        done, error = False, repr(e)[:200]
    res = rl.shelved(rob.m, rob.d)
    L = rec.log
    phase = np.array([r["phase"] for r in L.rows])
    held = np.array([r["slip_mm"] for r in L.rows]) > 0  # grasp_ref set: between grasp and release
    carried = held & np.array([not p.startswith(("shelve", "retreat", "done")) for p in phase])  # before set-down
    aborted = any("abort" in p for p in phase)
    book_z = rob.d.xpos[rob.m.body(rl.BOOK).id][2]
    outcome = ("shelved" if res["shelved"] else "put back" if aborted
               else "dropped" if book_z < 0.3 else "misplaced")
    normal = np.maximum(L.column("left_normal"), L.column("right_normal"))
    out = dict(cfg, outcome=outcome, completed=bool(done), error=error,
               slip_mm=float(L.column("slip_mm")[carried].max(initial=0.0)),
               turn_deg=float(L.column("slip_deg")[carried].max(initial=0.0)),
               squeeze_peak=float(normal.max()), squeeze_hold=float(np.median(normal[held])) if held.any() else 0.0,
               register_final=int(L.column("gripper_force")[-1]), time_s=float(L.column("t")[-1]),
               neighbours_mm=float(L.column("neighbours_mm").max()),
               err_mm=[float(v) for v in res["err_mm"]], tilt_deg=res["tilt_deg"], yaw_deg=res["yaw_deg"])
    print(f"{ctrl:11s} m={mass:.1f} mu={mu:.1f}: {outcome:9s} slip {out['slip_mm']:6.1f} mm "
          f"turn {out['turn_deg']:5.1f} deg squeeze peak {out['squeeze_peak']:6.1f} N hold {out['squeeze_hold']:6.1f} N "
          f"t {out['time_s']:5.1f} s {error}", flush=True)
    return out


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"skin": "#d62728", "fixed-10N": "#9ecae1", "fixed-40N": "#4292c6", "fixed-70N": "#2171b5",
              "fixed-125N": "#08306b"}
    marker = {"shelved": "o", "put back": "s", "dropped": "x", "misplaced": "^"}
    fig, axes = plt.subplots(2, len(FRICTIONS), figsize=(5.2 * len(FRICTIONS), 6.4), sharex=True)
    for k, mu in enumerate(FRICTIONS):
        for ctrl in CONTROLLERS:
            rs = sorted((r for r in results if r["controller"] == ctrl and r["mu"] == mu), key=lambda r: r["mass"])
            if not rs:
                continue
            m = [r["mass"] for r in rs]
            off = 0.03 * (CONTROLLERS.index(ctrl) - 2)
            axes[0, k].plot([v + off for v in m], [r["squeeze_hold"] for r in rs], "-", color=colors[ctrl], lw=1.2,
                            label=ctrl)
            axes[1, k].plot([v + off for v in m], [max(r["slip_mm"], 0.05) for r in rs], "-", color=colors[ctrl], lw=1.2)
            for r in rs:
                axes[0, k].plot(r["mass"] + off, r["squeeze_hold"], marker[r["outcome"]], color=colors[ctrl], ms=6,
                                mfc="none" if r["outcome"] != "shelved" else colors[ctrl])
                axes[1, k].plot(r["mass"] + off, max(r["slip_mm"], 0.05), marker[r["outcome"]], color=colors[ctrl],
                                ms=6, mfc="none" if r["outcome"] != "shelved" else colors[ctrl])
        axes[0, k].set_title(f"true PDMS-cover friction mu = {mu:g} (skin assumes {MU_ASSUMED:g})", fontsize=9)
        axes[0, k].set_ylabel("squeeze while holding (N per blade)")
        axes[1, k].set_ylabel("book slip in the grasp (mm)")
        axes[1, k].set_yscale("log")
        axes[1, k].set_xlabel("book mass (kg)")
        for a in axes[:, k]:
            a.grid(alpha=0.3)
    h = [plt.Line2D([], [], color=colors[c], lw=2, label=c) for c in CONTROLLERS]
    h += [plt.Line2D([], [], color="k", marker=mk, ls="none", mfc="none" if o != "shelved" else "k", label=o)
          for o, mk in marker.items()]
    fig.legend(handles=h, loc="upper center", ncol=9, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(40, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablation")
    ap.add_argument("--figure", type=Path, default=ROOT / "docs" / "figures" / "sim_ablation.png")
    args = ap.parse_args()
    grid = [dict(controller=c, mass=m, mu=f) for c, m, f in itertools.product(CONTROLLERS, MASSES, FRICTIONS)]
    if args.quick:
        grid = [g for g in grid if g["mass"] in (0.8, 2.4) and g["mu"] == 1.0 and g["controller"] in
                ("skin", "fixed-10N", "fixed-125N")]
    with get_context("fork").Pool(min(args.jobs, len(grid))) as pool:
        results = pool.map(one, grid, chunksize=1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plot(results, args.figure)
    print(f"{len(results)} runs -> {args.out / 'results.json'}, {args.figure}")


if __name__ == "__main__":
    main()
