# Robot mount: why the dof="full" PDMS flex diverges on the UR10e + 2F-140, and how to run it

MuJoCo 3.17.0 (python wheel, plus an instrumented copy of the 3.17.0 source, commit 800a109a in build/SOURCE_COMMIT.txt).
All paths below are relative to `docs/report_flex/study/robot/`.
How we detect divergence: the `mjWARN_BADQACC` counter (MuJoCo resets silently after a NaN), plus fatal arena overflows
(`mj_stackAlloc ... PrimalAllocate`), which are the same blow-up showing up as 20-100k contacts.

## 1. Root cause of F4: the island constraint solver leaves out the flex stiffness (engine bug)

**What it is.** With `integrator="discrete"`, `qacc_smooth` comes from the implicit effective metric
`M + hD + h^2 K` (mj_effSolve, monolithic). The CG constraint solver then runs **per island** and uses the island
metric operator `mj_effMulAddIsland`. That operator decides whether a flex's stiffness is applied
matrix-free by calling `flexMatrixFree()` (src/engine/engine_metric.c:2485-2491):

```c
return mjd_flexStiff_active(m, f, 1, 1) && (!d->nefmK || !mj_flexSimple(m, f));
```

`mjd_flexStiff_active` (src/engine/engine_derivative.c:1901-1913) is the gate for CSR assembly, and it returns 0 for
any flex that is not `mj_flexSimple`. So a flex with general (articulated) attachments is **not assembled into the CSR
and not applied matrix-free either**: its stiffness is simply missing from the island metric. The monolithic operator
`mj_effMulAdd` (engine_metric.c:236-246) handles exactly this case (`if (!d->nefmK || !mj_flexSimple(m, f))
mjd_flexBend/Stretch_mulRange`), so the two operators disagree.

**Why our model hits it.** The back face of the PDMS is pinned to `left_pdms`, and that body's weld ancestor is a
finger link with hinge dofs, so `mj_flexSimple` = 0 (`nefmdof` = 0, i.e. no CSR). The island path only runs when there
is at least one constraint row. That is why "all joints locked" and `lock_grip` (no limit or equality rows) are stable,
and why every robot configuration that has a constraint (the 2F-140 mimic equalities, an active limit, a contact)
diverges. "An upstream joint is free" was a stand-in for "the flex is non-simple and sits in an island". It was not a
conditioning effect.

**Evidence (all verified by runs):**
| check | result | script / output |
|---|---|---|
| engine operators vs explicit metric, minimal model (3x3x2 flex on 1 hinge + 1 joint equality) | monolithic operator = h^2 K_fd on the flex block to 9e-9 (rel.). **Island operator flex block = 0 exactly.** Largest eigenvalue of (M+S_isl)^-1 (M+S_mono) = **3598** (1.000 with the fix) | `metric.py`, `out/metric_min3x3x2_fix{0,1}.json` |
| same, minimal 11x15x3 | island flex block 0, max gen. eig **1.5e4**, fix -> 1 | `out/metric_min11x15x3_fix{0,1}.json` |
| same, full robot at home pose (arm, gripper, box dofs free; nv 1998, island nv 1992) | island flex block 0, max gen. eig **1.50e4**; fix -> 1.000; monolithic vs FD 4.4e-7 | `out/metric_robot_fix{0,1}.json` |
| is the metric positive definite with arm joints free? | yes: A_ref = M+hD+h^2K (FD, all dofs incl. arm, actuator kp/kv, joint damping), M+S_mono and M+S_isl all have 0 non-positive eigenvalues | same files, `eig_*` |
| does mj_effSolve take the silent pAp <= 0 exit (F4)? | **no**: negcurv = 0 in every call; it hits the 100-iteration cap at relres 1e-8 to 3e-6 (that is the "ill-conditioned" warning, harmless here) | instrumented build, `MJ_EFFDBG=1`, `out/effdbg_robot_fix{0,1}.txt` |
| first steps on the robot | flex max qacc at step 1: 7.95 (bug) vs 0.455 (fix); then x1.8e3 and x5.7e5 per step, NaN at step 3-4. The island CG converges in 6-23 iterations without the fix (it is solving with plain M) and in 46-100 with the fix | `out/effdbg_robot_fix{0,1}.txt` |
| switch only the island path | `<flag island="disable"/>` (monolithic constraint solve): stable in every configuration tried; engine patch (below, env `MJ_FIX_ISLANDFLEX=1`): stable | `out/s1.jsonl`, `out/s2.jsonl`, `out/s3.jsonl`, `out/m1.jsonl` |

**Mechanism.** The constraint solver computes the constraint correction using a metric about 1.5e4 times too soft in
the flex directions. That has the same effect as integrating the PDMS stiffness explicitly with h*omega of about
sqrt(1.5e4) = 120, far beyond the stability limit, so the error grows by orders of magnitude every step.

**Engine fix** (`fix/island_flex_fix.patch`, against 3.17.0 engine_metric.c `flexMatrixFree`): select flexes
the same way as the monolithic `mj_effMulAdd`:
```c
if (m->flex_rigid[f] || (m->flex_dim[f] == 1 && m->flex_bendingadr[f] < 0)) return 0;
return mj_effFlexStiffPossible(m, f) && (!d->nefmK || !mj_flexSimple(m, f));
```
(The study build has the same change behind an env switch: `build/me295_instrument_metric.patch`, library in
`build/bld/lib`, python package copy in `pymj/` (use with `PYTHONPATH=pymj MJ_FIX_ISLANDFLEX=1`).) One difference
remains: the monolithic path also applies damping-only flexes, while this gate needs stiffness. We did not test
damping-only flexes. Not posted upstream.

### Minimal repro
`repro/minimal_repro.xml` contains one hinge (armature 0.1, damping 2, like the UR10e size2 joints), a capsule link, a
child body (mass 1e-6) and a pinned 3x3x2 dof="full" flex 51 x 71 x 1.5 mm, plus one `<joint>` equality. Run
`python repro/run_repro.py`: islands on -> BADQACC at step 3; islands off -> none in 200 steps (`out/repro_wheel.txt`).
With the patched library both are stable (`out/repro_pymj_fix.txt`). Variants (`minimal.py`, `out/m1.jsonl`): a 2x2x2
flex (nv 13) also diverges. So do an active joint limit instead of the equality, a floor contact, a slide joint
instead of the hinge, nu 0.3, E 1 kPa, h 0.1 ms, and parent mass 1 kg. No constraint row: stable. No joint (flex
simple): stable.

### Checklist items (task (1)-(5))
1. **Material / conditioning:** raw (1 MPa, 0.45), F1-corrected (971787 Pa, 0.409091), nu 0.3 raw and F1, F1 nu 0.49,
   E 1e5 and 1e4: all diverge at step 3-5 with islands on, and all are stable with islands off (`out/s1.jsonl`,
   `out/s2.jsonl`). Conditioning is not the cause.
2. **Metric PD:** see the table above. It is PD, and the solve never takes the pAp <= 0 exit.
3. **Articulated-attachment path:** our flex takes the general path (`mj_flexSimple` = 0, `nefmdof` = 0, matrix-free
   `mjd_flexStretch_mulRange` through `flexMap`). For dim=3 StVK at the operating state, that operator equals the
   finite-difference stiffness: flex block relerr 1e-8 (minimal) and 4e-7 (robot). The flex-to-arm coupling block is
   4.7e-10 to 3.8e-6 (FD: 1e-11), negligible against the flex block (about 1.6-2.6). The path is correct in the
   monolithic solve and missing in the island solve. The tiny parent inertia is not the cause: pdms_mass 1e-2 (robot)
   and 1 kg (minimal) still diverge. The 2F-140 equalities are not the cause as such: without them (`noeq`) the robot
   still diverges, because limit rows still form an island.
4. **Solver options (islands on, robot):** CG iterations 1000, tol 1e-12, h 1e-4 / 2e-4 / 1e-3, impratio 1, pyramidal
   cone, no warmstart and arena 2 GB all diverge at step 3-4 (`out/s1.jsonl`). The engine refuses Newton
   ("discrete integrator: flex with general attachments requires solver='CG'") and noslip/PGS ("PGS and noslip not yet
   supported with flex"). In the minimal model (`integ.py`, `out/integ_min3x3x2.txt`): Euler diverges with islands on
   or off (explicit elasticity). implicit/implicitfast are refused for flex elasticity in 3.17. RK4 showed no BADQACC
   in 200 steps (not studied further).
5. **Bisection:** the smallest diverging model is the 2x2x2 flex on one hinge plus one constraint row (above). On the
   robot, removing actuators, gravity-off (trivially stable: nothing moves), armature 0, one flex only, coarser or finer
   grids (6x8x2 ... 21x29x3), locking the arm, or locking all but wrist_3 do not help as long as a constraint row
   exists. Not run: `damping0`, whose spec edit failed with a TypeError in run.py (`out/s1.jsonl`).

## 2. A second, independent problem: the contact settings collapse the layer

Once the divergence is removed (islands off or the patch), the grasp ran without NaN. But the box was held mainly by
the **rigid blade rims**, and the PDMS front layer **inverted** (`out/g1.jsonl`, `out/diag_*.txt`,
`out/diag2_noisl_f1*.txt`): about 850 of 1680 tets inverted, the free face pushed about 1.5-3 mm through the 1.5 mm
layer, and box forces not in balance. Fine trace (`contact_trace.py`, `out/trace_noisl_f1_align.txt`): at the first
face-on contact, 275 flex contacts per side; the normal force climbs to **about 1000 N per side within 25 ms** (the
2F-140 can only squeeze about 120 N), the compression reaches 0.5 mm, and then the layer snaps through.

**Why the layer can invert at all** (`mirror_test.py`, `out/mirror_test.json`): the full flex elasticity depends only
on edge lengths (F1: energy built from squared edge-length changes), so it is blind to reflection. Push the free face
through the thickness and the restoring force peaks at w = 0.3 mm (2637 N for the whole face, F1 inputs; front
sub-layer 0.75 mm thick). It falls to 0 when the elements are flat (w = 0.75 mm), changes sign beyond that, and is
exactly **0 in the mirrored state** (w = 1.5 mm). There is no inversion barrier, so a transient overload leaves the
layer inside-out for good.

**Why the overload happens:** the flex contact `solref="0.0001 1"` (time constant h/5, `refsafe` disabled) plus the
CG cap of 100 iterations. Sweep on 6x8x2 (`out/c1.jsonl`, `c1summ.py`; islands off, F1, both blades):
| flex contact | held N_L / N_R (N) | rigid blade (N) | PDMS compression | inverted tets | max N_L over run |
|---|---|---|---|---|---|
| solref 1e-4 (current), it 100 | 26 / 13 | 205 | 3.1 mm | 210 / 205 | 65764 N |
| solref 1e-4, it 1000, tol 1e-10 | 120.6 / 120.8 | 0 | 43 um | 0 | 124 |
| solref 5e-4 (= h) | 194 +- 84 / 197 +- 76 | 0 | 3.0 mm | 9 / 1 | 924 |
| solref 1e-3 (2h) | 120.6 / 120.8 | 0 | 26 um | 0 | 124 |
| solref 2e-3 (4h) | 119.1 / 119.0 | 3.4 | 22 um | 0 | 120 |
| solref 5e-3 | 57 / 57 | 127 | 13 um | 0 | 58 |
| solref 2e-2 | 6 / 6 | 229 | 1 um | 0 | 6 |

A soft time constant brings a second effect: contact penetration. Under a given force it grows roughly with
tc^2 (1-d)/(d m_vertex), and the vertices weigh only about 10 mg (11x15x3). On 11x15x3 with solref 1e-3 and solimp
0.99 the largest flex-contact penetration is **0.62 mm**, more than the 0.5 mm the PDMS stands proud, so the rigid
blade takes 9 N (`out/p1.jsonl`). solimp 0.9999 cuts (1-d)/d by a factor of 100 (see section 3).

## 3. Workaround that runs on the robot, with proof

**Recommended settings (no engine change):** add `<flag island="disable"/>` to the robot `<option>`, use the F1-corrected
material inputs (young 971787 Pa, poisson 0.409091 for nu 0.45; 993552 / 0.480392 for nu 0.49), and choose one of the
two contact settings below. The alternative to `island="disable"` is the engine patch (`fix/island_flex_fix.patch`),
which gives the same results (rows marked "patch").

Proof runs: `grasp.py`, launched through `sweep.py`, single thread, from `out/p1_opts.json` and `out/q1_opts.json`.
Results are in `out/p1.jsonl` and `out/q1.jsonl`, time series in `out/p_*.npz`, summaries from `c1summ.py`. The
sequence is sim_gripper's: home pose, box (300 g, 50 x 80 x 60 mm) on the pedestal, 0.2 s settle, close at
150 mm/s for 1.2 s, IK-lift 50 mm in 1.0 s, hold 0.3 s. That is **2.7 s** of simulation with the full flex on **both**
blades. Each row is one run. "held" means the mean +- std over the last 0.1 s of the hold. "rigid" is the normal force
the box gets from the rigid blade geoms. "comp" is the largest through-thickness displacement of the PDMS free face.
"pen" is the largest flex-contact penetration.

| run (11x15x3 unless noted) | BADQACC | held N_L / N_R (N) | rigid (N) | comp L/R (um) | inverted tets | pen (mm) | box lift (of 50 mm) | ms/step all / in contact |
|---|---|---|---|---|---|---|---|---|
| islands off, contact solref 1e-4 (current XML) | 0 | 45 +- 38 / 28 +- 59 | 158 | 1631 / 2972 | 798 / 866 | 0.03 | 50.3 | 24.1 / 31.4 |
| **A: islands off, solref 1e-4, iterations 1000, tol 1e-10** | 0 | **120.2 +- 1.7 / 120.1 +- 1.5** | **0** | 64 / 57 | 0 | **0.055** | 49.2 | **144.9 / 252.9** |
| **B: islands off, solref 1e-3 (2h)** | 0 | **121.2 +- 0.2 / 121.9 +- 0.2** | 9.1 | 45 / 46 | 0 | **0.62** | 48.7 | **27.7 / 38.0** |
| B with the box yawed onto the PDMS normal | 0 | 121.9 +- 0.4 / 121.2 +- 0.3 | 9.1 | 48 / 48 | 0 | 0.63 | 48.8 | 27.6 / 37.8 |
| B, F1 nu 0.49 | 0 | 125.3 +- 2.4 / 126.7 +- 3.0 | 0 | 38 / 27 | 0 | 0.45 | 50.0 | 32.2 / 42.8 |
| B, **patch** (islands on, MJ_FIX_ISLANDFLEX=1) | 0 | 120.2 +- 0.4 / 119.8 +- 0.2 | 9.1 | 45 / 47 | 0 | 0.63 | 48.5 | 30.9 / 42.2 |
| B, islands on, stock MuJoCo | fatal at step 3 (arena overflow) | - | - | - | - | - | - | - |
| B, 6x8x2 | 0 | 119.7 +- 3.9 / 119.1 +- 3.5 | 0 | 29 / 30 | 0 | 0.17 | 49.9 | 6.1 / 10.4 |
| B, 21x29x3 | 0 | 118.5 +- 0.5 / 118.8 +- 0.5 | 7.4 | 67 / 68 | 0 | 0.60 | 48.8 | 105.2 / 141.7 |
| solref 2e-3 (4h) | 0 | 89.3 / 87.6 | 64.4 | 28 / 30 | 0 | 1.04 | 50.3 | 29.6 / 41.6 |
| solref 1e-3 + solimp 0.9999 (and 0.999; also on 21x29x3, nu 0.49, patch) | 0 | 23 +- 7 / 2 +- 7 | 190 | about 3000 | 900 / 972 | 0.08 | 46.5 | 23.9 / 30.8 |

What the table shows:
- In every configuration with islands off or the patch, there is no NaN or BADQACC over 2.7 s, and the box is lifted
  and held (48.5-50 mm of the 50 mm command).
- **Setting A** gives sensible contact. Both sides carry 120 N, which is the 2F-140's squeeze (the model's comment
  says about 120-128 N per finger at forcerange 24 N m), equal on left and right within 0.1 N. The rigid blade carries
  nothing, no element inverts, the PDMS compresses about 60 um, and contact penetration stays at or below 55 um. The
  cost is about 5x setting B (145 vs 28 ms per 0.5 ms step, single thread): 2.7 s of simulation took 932 s.
- **Setting B** is cheap (about 15% over the old stable-but-wrong runs) and gives balanced, steady normal forces, but
  the contact model, not the PDMS, provides most of the compliance. Penetration of up to 0.62 mm (vertex mass about
  10 mg and a soft time constant) exceeds both the PDMS deformation (45 um) and the 0.5 mm the pad stands proud, so 7%
  of the load goes through the blade rims. Taxel *forces* are usable (left skin total 115 N); taxel *deformations*
  and the layer's stiffness are not. Penetration falls with coarser grids (0.17 mm on 6x8x2, heavier vertices).
- Making the contact harder through the impedance (solimp 0.999 / 0.9999) brings back the overload-and-invert failure
  at the 100-iteration cap. The stiffness the 100-iteration CG can resolve is the real limit.
- Aligning the box with the PDMS normal makes no difference in B. The 0.48 deg yaw that sim_gripper leaves
  (`out/diag2_noisl_f1.txt`) is not what drives the collapse.

**Cost per step** (11x15x3, both flexes, one thread, contact phase): B 38 ms (islands off) or 42 ms (patch); A 253 ms.
Over the whole run: B 27.7 ms (55x slower than real time at h = 0.5 ms), A 145 ms.

**Rules that follow** for running a full flex on the robot with the discrete integrator in 3.17.0:
(1) islands off, or the patch; (2) F1-corrected inputs; (3) either keep the stiff flex contact (solref 1e-4) and give
the CG enough iterations to converge it (1000 / 1e-10 here), or accept contact compliance that is larger than the
PDMS's when using solref of at least 2h; (4) watch the inverted-tet count (`grasp.py` logs invL/invR). Once the layer
inverts it does not recover, because the StVK edge-length energy is zero for mirrored elements.

## 4. What is verified and what is assumed
- Verified by runs: everything in the tables (files given). Verified in source: the two selection rules
  (engine_metric.c 236-246 and 2485-2491, engine_derivative.c 1901-1913, engine_core_util.c 1277-1283 / 1733-1764).
- Assumed or not done: no upstream post. The patch is only tested on these models: dim=3 and dim=2 not separately, no
  damping-only flex, no interpolated flex. RK4 was not examined. `damping0` bisection was not run (script error).
  "120 N per finger" is the model's own comment (ur10e_2f140.xml line 360); the measured held normal force matches it.
