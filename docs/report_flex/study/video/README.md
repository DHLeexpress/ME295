# contact-video: close-up of the PDMS layer deforming under contact (MuJoCo 3.17 flex)

Deliverables (this directory, `docs/report_flex/study/video/`):

- `flex_contact_closeup.mp4`: 1920 x 1080, 30 fps, H.264 yuv420p, 1250 frames, 41.67 s (checked with ffprobe). It has a 3 s title card listing every object, then 8 segments of 130 frames each, each followed by a 0.5 s hold.
- `stills.png`: one composite frame per object, taken at the end of the first hold (full normal load, before the slide). 2 x 4 tiles at half resolution.
- `profile.png`: the "+1, +2, +1" picture. It shows u_z along a vertex grid line across three taxels, with the taxel averages and the taxel forces (sphere and cube, end of the hold).
- `prod2/seg<k>.{xml,json,npz}`: production simulations (scene, settings plus metrics, time series plus frames). `prod2/checks.json` holds the physics checks and `prod2/scales.json` the shared colour scales.

## Scene and settings (all from `sim.py`; every run writes its own XML in `prod2/seg<k>.xml`)

- **Pad:** 51 x 71 x 1.5 mm, `flexcomp type="grid" dim=3`, dof="full" (linear tets), 40 x 56 x 3 cells (41 x 57 x 4 vertices; cells 1.275 x 1.268 x 0.5 mm). Back face pinned with `<pin gridrange>`. The pad is fixed in space with no robot. The blade is drawn as a visual box. The 5 x 7 taxel borders are drawn on the deformed free face in the MuJoCo render.
- **Material:** target E = 1 MPa, nu = 0.45. The XML uses the F1-corrected inputs `young = 971786.8 Pa, poisson = 0.409091`. Density 965 kg/m^3 (flex mass 5.24 g). `damping = 0`, with no viscoelasticity and no Rayleigh damping. The only dissipation is the numerical dissipation of the `discrete` integrator.
- **Solver and contact:** timestep 0.5 ms, `integrator="discrete"`, CG with 3000 iterations, tolerance 1e-8, elliptic cone, impratio 100, refsafe disabled, gravity 0. Contact `solref="3e-06 1"`, `solimp="0.9999 0.9999 0.001 0.5 2"`, mu = 1 (condim 3), radius 0, no margin.
- **Objects:** 8 rigid objects, generated pseudo-randomly from seed 20261030 (`sim.objects()`):

  | # | object | start (x, y) mm |
  |---|---|---|
  | 1 | sphere R 7.7 mm | (3.2, -13.3) |
  | 2 | rod lying across, R 3.8 mm, yaw 118 deg | (-9.0, -7.7) |
  | 3 | box edge, 15 mm, yaw 138 deg | (-3.0, 11.4) |
  | 4 | book corner 40 x 30 x 12 mm, yaw 141 deg | (6.0, 17.1) |
  | 5 | capsule tip R 3.6 mm, tilt 28 deg | (0.7, 5.3) |
  | 6 | ellipsoid 25 x 8 x 8 mm, random orientation | (-10.7, -7.1) |
  | 7 | coin R 10.6 mm, 1.8 mm thick, tilt 3.9 deg | (10.9, -10.0) |
  | 8 | 8 mm cube, yaw 69 deg, tilt 1.7 deg | (12.7, 6.6) |

  Slide distances (2.1-4.0 mm) and directions are listed on the title card.
- **Drive:** each object has 3 slide joints with position servos (kp 1e10, dampratio 1). A mocap body cannot apply friction (bench_dynamic finding 7).
- **Timeline per segment:** approach 80 ms (from a 1.5 mm gap to 20 um), press 400 ms, hold 80 ms, slide 400 ms, hold 80 ms, lift 120 ms. That is 2320 steps; a frame is stored every 18 steps (9 ms), giving 130 frames. Playback runs at 0.27 s of simulated time per second of video.
- **Press stop:** pressing stops at the target load (5-20 N, drawn at random) or when the object has travelled `DEPTH_CAP = 100 um` past first touch, whichever comes first.

## Results per segment (`prod2/checks.json`, written by `analyze.py`)

| # | object | target N | reached Fz max N | Ft max N | travel um | max u_z um* | pen_fw at hold um (share) | max contacts | min tet J | NaN | sim wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sphere R 7.7 | 8.8 | 0.88 (cap) | 0.67 | 97 | 130 | 14.4 (0.15) | 7 | 0.886 | none | 176 |
| 2 | rod R 3.8 | 5.3 | 6.51 (force) | 6.45 | 25 | 48 | 2.8 (0.11) | 138 | 0.946 | none | 951 |
| 3 | box edge 15 mm | 10.2 | 4.21 (cap) | 3.99 | 97 | 149 | 15.5 (0.16) | 21 | 0.823 | none | 817 |
| 4 | book corner | 9.5 | 0.89 (cap) | 1.08 | 97 | 145 | 28.9 (0.30) | 2 | 0.795 | none | 132 |
| 5 | capsule tip R 3.6 | 8.3 | 0.82 (cap) | 0.78 | 97 | 114 | 17.9 (0.18) | 5 | 0.754 | none | 148 |
| 6 | ellipsoid | 5.6 | 0.73 (cap) | 0.77 | 97 | 116 | 28.9 (0.30) | 6 | 0.766 | none | 140 |
| 7 | coin R 10.6 | 18.7 | 2.32 (cap) | 2.11 | 97 | 139 | 12.3 (0.13) | 31 | 0.855 | none | 768 |
| 8 | cube 8 mm | 5.7 | 4.19 (cap) | 2.62 | 97 | 145 | 14.9 (0.15) | 51 | 0.808 | none | 1145 |

Notes on the columns:

- **max u_z\*** is the largest free-face u_z over the whole segment, including the slide. It includes contact penetration: vertices sit inside the object by up to `pen_max`. During the slide, u_z exceeds the object's travel by up to 16-52 um (`uz_minus_travel_slide_max_um`).
- **pen_fw** is the force-weighted contact penetration at the end of the hold. Share = pen_fw / mean u_z under the contact.
- **min tet J** is the minimum tet volume ratio. No tet inverted in prod2 (`ninv_max = 0`).
- No CNSTRFULL, CONTACTFULL or BADQACC warnings occurred, and CG never hit its 3000-iteration cap (`frac_steps_iter_cap = 0` in every segment).

### Physics checks (verified, `prod2/checks.json`, `analyze.py`)

- **Total contact force = sum of taxel forces:** the largest error over all steps and segments is 5.3e-15 N. The taxels bin the same per-contact forces, so this checks completeness only, not physics. Servo force vs contact force at the hold differs by 1e-4 to 0.12 N; it is largest for the cube, which rings.
- **Continuity across taxel borders:** the displacement is one C0 vertex field, and taxel borders are not element boundaries in the field. The largest u_z jump between neighbouring vertices across a border edge (`border_jump_um`) is the same size as the largest interior jump (`interior_jump_um`), for example 49.8 vs 99.6 um for the cube. These jumps are mesh-scale gradients under the indenter, not discontinuities.
- **`profile.png`, cube at the end of the hold:** taxel (3,3) F_n = 3.18 N, mean u_z 14.1 um. Its neighbour (3,4) has F_n = 0.61 N, mean u_z 3.8 um, and (3,2) has F_n = 0, mean u_z -0.1 um. Force is binned by contact point, so a taxel gets force only if a contact point lies inside it. Displacement spreads continuously into the neighbours. This is consistent with bench_dynamic finding 11.
- **Lift-off:** residual u_z after lift-off is at most 1.2e-4 um. The layer returns to rest, with no inverted tets left behind.
- **NaN:** none in any prod2 run (`nan_step = null`).

### What the video shows, and its limitations (stated plainly)

1. **The requested loads of 5-20 N are NOT reached by curved or sharp objects.** At the 100 um indentation cap, the sphere, corner, capsule and ellipsoid give 0.7-0.9 N, and the coin, box edge and cube give 2.3-4.2 N. Only the rod (a line contact, 25 um) reached its target. The task text's "5-20 N, i.e. tens of um" does not hold for this 1 MPa, 1.5 mm layer: tens of um gives about 1 N for a 4-12 mm sphere.
   - Deeper indentation was tried in the first production run (`runs/`, cap 250 um). The corner, capsule and ellipsoid inverted 6 surface tets during the slide (J about -0.9, found with `inv_check.py`), and these stayed inverted after lift-off, with u_z of 830 um at zero force.
   - The cause is the energy: it is a function of squared edge-length changes, which is reflection-invariant, so mirrored tets are stress-free.
   - The cap was therefore lowered to 100 um (`prod2/`), which keeps the strains small (travel / T = 6.5 %).
2. **Contact penetration is NOT << deformation.** It is 11-30 % of u_z at the hold (column pen_fw), and up to 99 um at single vertices (box edge, during the slide).
   - Lowering solref from 1e-5 to 3e-6 or 1e-6 did not lower it further: ellipsoid pen_fw was 14.9 / 14.8 um (`test/sr3e-6`, `test/sr1e-6`, `logs/t4_*.log`).
   - Raising the flex mass x10, which helped in bench_dynamic C1f, became unstable here. For the sphere, `logs/t2_s0_m10.log` shows u_z = 0 at Fz max with pen 261 um, meaning the object went through the layer. For the coin, `logs/t2_s6_m10.log` shows pen 231 um and u_z max 734 um.
   - This matches bench_dynamic finding 1: solref stiffness scales with the mass of the touching vertices.
3. **Local contacts are under-resolved at 1.27 mm cells.**
   - The sphere's contact radius is about sqrt(2 R d) = 1.2 mm, about one cell, so its dimple is one or two vertices wide (`profile.png`).
   - The rod and box-edge trenches show a mesh-scale sawtooth along the line.
   - The book corner touches with 1-2 contacts and hops from vertex to vertex while sliding. Fz drops to about 0 four times (visible in its time strip).
   - These are discretisation artefacts, not physics (cf. bench_dynamic finding 12).
4. **Ringing and chatter: no material damping.**
   - During the hold, the std of max u_z is 0.03-1.4 um for most objects. For the cube it is 6.6 um (13.5 um peak-to-peak), and its Fz std is 0.043 N.
   - During the slide, Fz std is 0.07-0.41 N because of stick-slip and contact switching.
   - Computed by `ringing.py` from `prod2/seg<k>.npz`; output `prod2/ringing.json`.
5. **The pad-frame |F_t| / F_z exceeds mu = 1 during some slides:** 3.09 for the book corner, 1.51 for the box edge. The likely cause is that the contact normals are not the pad normal: a corner or edge pressing into a pit has normals tilted toward the box faces, so the normal force has a pad-tangential component. Verified with `cone_check.py`, which re-runs segment 4 with the prod2 settings up to step 1620, covering the worst steps (output `test/cone/cone.json`):
   - The re-run reproduces prod2: Fz max 0.890 N and Ft max 1.081 N.
   - In the contact frame, max |f_t| / f_n = 1.0000 over 1320 contact steps, so every contact stays inside its friction cone.
   - Contact normals tilt up to 74.8 deg from the pad normal.
   - A pad-frame ratio of 3.09 is therefore a geometric effect (tilted normals), not a friction-cone violation.
   - The same result holds with solref 1e-5 (`test/cone_solref1e-5/cone.json`).
6. **`max u_z` in the strip** is the instantaneous maximum over the free face, so it includes penetration and ringing. It is drawn as a thin grey line.

## Visualisation (`render.py`)

Each frame is a matplotlib composite at 1920 x 1080, with a header (object, time, phase, material, mesh, target load / cap) and four panels:

1. MuJoCo offscreen render (EGL), TRUE scale, close-up camera, with the taxel borders drawn as thin capsules on the deformed face.
2. The free face in a 41 x 31 mm window around the object's mid-slide point, drawn as a 3D surface.
   - Displacement (all three components) is EXAGGERATED x40, and the factor is printed on the panel. x40 was chosen because max u_z is about 100-150 um; x100 made 10-15 mm pits that hid the shape.
   - Facets are coloured by the true u_z in um (diverging colour bar, + = into the layer) and get Lambert shading from the exaggerated geometry.
3. 5 x 7 taxel normal force (N, colour and numbers) with shear arrows (scale on the panel), the sum of the taxels, and the contact count.
4. Time strip: total normal F_z, tangential |F_t|, object travel past first touch, and max u_z, with phase bands and a cursor.

An exaggerated MuJoCo render (the optional item) was NOT made.

## How to regenerate

```
cd docs/report_flex/study/video
P=/home/dohyun/miniforge3/bin/python
# 1) simulate the 8 segments in parallel (one process each; writes prod2/seg<k>.*)
for k in 0 1 2 3 4 5 6 7; do $P sim.py $k --out prod2 > logs/prod2_seg$k.log 2>&1 & done; wait
# 2) checks + profile plot
RUNS=prod2 $P analyze.py
# 3) frames (90 processes over GPUs 0-3), video and stills
RUNS=prod2 EXAG=40 NPROC=90 GPUS="0 1 2 3" ./render_all.sh
```

`sim.py` defaults must match prod2: DEPTH_CAP 0.10e-3 and solref 3e-6. Check `prod2/seg0.json` (solref, depth_cap_m) after a rerun.

## Timing (measured)

- Simulation, 8 processes in parallel, 2320 steps each: total wall 132-1145 s per segment (`wall_total_s`). Median 62-71 ms/step for 1-7 contacts and 380-530 ms/step for 21-138 contacts (`wall_median_ms`). The CG iteration count dominates; for the cube, iterations reach 1500 per step.
- Rendering: 1040 segment frames in about 40 s on 88 processes over 4 GPUs (frame file timestamps). Each process spends about 10 s on model compile and EGL setup. Encoding with ffmpeg libx264 (preset slow, crf 18) takes about 10 s.

## Earlier attempts kept here (not deliverables)

- `runs/`: first production with a 250 um cap and solref 1e-5. It produced the tet inversions described above (`inv_check.py`).
- `prod/`: 100 um cap, solref 1e-5 (`prod/checks.json`).
- `test/*`, `logs/t*`: mesh, mass, iteration and solref tests. In `logs/t3_pgs.log`, PGS fails to compile with "discrete integrator: PGS and noslip not yet supported with flex". In `logs/t3_it3000.log`, CG 1000 iterations hit the cap on 46 % of the coin's steps, while 3000 iterations converge.
- `render_v1_backup.py`: the earlier draft of the renderer (x100, other view). `frames_test/` holds test frames.

## Verified vs assumed

- **Verified by runs in this directory:** every number in the tables, the force-sum identity, no NaN, no inverted tets in prod2, per-contact friction cones (segment 4), determinism (the segment 4 re-run reproduces prod2), and the video format.
- **Assumed / quoted:**
  - The F1 correction is taken from the main session's patch tests (not re-run here).
  - The mass-scaling explanation of the penetration share comes from bench_dynamic finding 1.
  - "Numerical dissipation of the discrete integrator" was not quantified.
