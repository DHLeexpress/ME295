"""Run plans for sweep2.py (compare task). All runs: rig2.py, nu = 0.45 unless stated, hold 0.2 s after a 10 ms ramp.

sd     static-vs-dynamic reconciliation: C1f, mocap driver, box vs plane indenter, stiff vs stiffest, mass x1 vs x100.
robot  candidate cell structures for the robot (full_corr only), robot options, 3 contact settings, 8 cases,
       5 lateral placements for the local cases (C4, C5, C6, C6f) -> placement confidence intervals.
rig    candidate cell structures for the fixed-rig sensor study (full / trilinear / quadratic).
nu49   the chosen structures at nu = 0.49.
"""
import itertools

CASES_GLOBAL = ["C1", "C1f", "C2", "C3"]
CASES_LOCAL = ["C4", "C5", "C6", "C6f"]
# fixed pseudo-random placements (fractions of the in-plane element size); the first is the centred one
OFFS = [(0.0, 0.0), (0.31, 0.17), (-0.23, 0.41), (0.44, -0.36), (-0.12, -0.27)]
BASE = dict(hold_s=0.2, nu=0.45)


def elem_size(variant, cells):
    k = 2 if variant == "quadratic" else 1
    return 0.051 / (k * cells[0]), 0.071 / (k * cells[1])


def combo(variant, cells, contact, iterations, tol=1e-8, mass_scale=1.0, split=1, nu=0.45, offsets=True, extra=None,
          cases=None):
    out = []
    dx, dy = elem_size(variant, cells)
    for case in (cases or CASES_GLOBAL + CASES_LOCAL):
        offs = OFFS if (offsets and case in CASES_LOCAL) else OFFS[:1]
        for fx, fy in offs:
            c = dict(BASE, variant=variant, cells=list(cells), case=case, contact=contact, iterations=iterations,
                     tolerance=tol, mass_scale=mass_scale, nu=nu)
            if split != 1 and case in ("C1", "C1f", "C2", "C6", "C6f"):
                c["split"] = split
            if (fx, fy) != (0.0, 0.0):
                c["offset_m"] = [round(fx * dx, 9), round(fy * dy, 9)]
            if extra:
                c.update(extra)
            out.append(c)
    return out


ROBOT_MESHES = [[5, 7, 1], [5, 7, 2], [10, 14, 1], [10, 14, 2]]
ROBOT_CONTACTS = [("robotB", 100, 1e-8), ("stiff", 100, 1e-8), ("stiff", 1000, 1e-10)]

RIG_COMBOS = [
    dict(variant="full_corr", cells=[20, 28, 2], contact="stiff", iterations=1000),
    dict(variant="full_corr", cells=[20, 28, 2], contact="stiffest", iterations=1000, mass_scale=100),
    dict(variant="full_corr", cells=[40, 56, 2], contact="stiff", iterations=1000),
    dict(variant="full_corr", cells=[40, 56, 2], contact="stiffest", iterations=1000, mass_scale=100),
    dict(variant="trilinear", cells=[20, 28, 2], contact="stiffest", iterations=1000, mass_scale=100, split=5),
    dict(variant="quadratic", cells=[10, 14, 1], contact="stiffest", iterations=1000, mass_scale=100, split=5),
    dict(variant="quadratic", cells=[20, 28, 1], contact="stiffest", iterations=1000, mass_scale=100, split=5),
]


def groups():
    G = {}
    sd = []
    for cells, ind, (contact, ms) in itertools.product([[10, 14, 2], [20, 28, 2], [20, 28, 3]], ["box", "plane"],
                                                      [("stiff", 1), ("stiffest", 1), ("stiff", 100),
                                                       ("stiffest", 100)]):
        c = dict(BASE, cells=cells, case="C1f", contact=contact, mass_scale=ms, iterations=1000, driver="mocap")
        if ind == "plane":
            c["indenter_type"] = "plane"
        sd.append(c)
    G["sd"] = sd
    G["robot"] = [c for m in ROBOT_MESHES for (ct, it, tol) in ROBOT_CONTACTS
                  for c in combo("full_corr", m, ct, it, tol)]
    G["rig"] = [c for rc in RIG_COMBOS for c in combo(**rc)]
    # robot meshes with mass-scaled contact (contact stiffness in N/m is proportional to the vertex mass) and an
    # intermediate time constant
    G["robotm"] = [c for m in ROBOT_MESHES for (ct, it, ms) in [("robotB", 100, 10), ("robotB", 100, 100),
                                                                 ("tc2e-4", 100, 1), ("robotB", 100, 1000)]
                   for c in combo("full_corr", m, ct, it, 1e-8, mass_scale=ms)]
    # graded tensor-product tet meshes (flexcomp type="direct"), refined around the indenter (fixed-rig option)
    G5a = dict(core=[0.0012, 0.0012], h0=0.00015, g=1.3, hmax=0.003, z=[0, 0.00015, 0.00035, 0.00065, 0.001, 0.0015])
    G5b = dict(core=[0.0010, 0.0010], h0=0.0002, g=1.4, hmax=0.005, z=[0, 0.0002, 0.0005, 0.0009, 0.0015])
    G6 = dict(core=[0.0060, 0.0060], h0=0.0005, g=1.4, hmax=0.005, z=[0, 0.0003, 0.0008, 0.0015])
    gr = []
    for gm, cases, nplace in ((G5a, ["C5"], 1), (G5b, ["C5"], 3), (G6, ["C4", "C6", "C6f"], 3)):
        for case in cases:
            for fx, fy in OFFS[:nplace]:
                c = dict(BASE, variant="full_corr", cells=[1, 1, 1], case=case, contact="stiff", iterations=1000,
                         graded=gm, max_step_s=10.0, timeout_s=14400)
                if (fx, fy) != (0.0, 0.0):
                    c["offset_m"] = [round(fx * gm["h0"], 9), round(fy * gm["h0"], 9)]
                gr.append(c)
    G["graded"] = gr
    # the chosen structures (and the current robot grid) at nu = 0.49
    G["nu49"] = (combo("full_corr", [5, 7, 1], "robotB", 100, nu=0.49)
                 + combo("full_corr", [10, 14, 2], "robotB", 100, nu=0.49)
                 + combo("full_corr", [40, 56, 2], "stiff", 1000, nu=0.49)
                 + combo("full_corr", [20, 28, 2], "stiff", 1000, nu=0.49))
    return G


if __name__ == "__main__":
    for k, v in groups().items():
        print(k, len(v))
