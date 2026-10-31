# PROGRESS (contact-video)
## Done
- sim.py: scene (pad 40x56x3 cells = 41x57x4 verts, dof=full, F1-corrected young 971787 / poisson 0.409091, back face
  pinned, mass x1, damping 0; solref 1e-5 1, solimp 0.9999.., mu 1, CG 3000 it, discrete, h 0.5 ms, elliptic, impratio 100);
  8 objects from seed 20261030 (objects() in sim.py); servo-driven object (3 slide joints, kp 1e10); press until
  F_target (5-20 N) or depth cap 0.25 mm; slide 2-4 mm; lift.
- Tests (test/, logs/t*_*.log): mass x10 went unstable (seg0 uz 0 at Fzmax, seg6 uz 734 um) -> rejected;
  PGS: "discrete integrator: PGS and noslip not yet supported with flex" (compile error);
  CG 1000 hits the cap on 46 % of coin steps, 3000 converges (same wall ~0.5 s/step);
  curved objects reach only 1.4-2.8 N at the 0.25 mm cap (loads 5-20 N need deeper indentation than linear range).
- Production sims launched: runs/seg<k>.{npz,json,xml}, logs/seg<k>.log
## Next
- render.py (per-frame composite), make_video.sh (ffmpeg), stills.png, profile plot, README.md
- FINDING: at 0.25 mm indentation cap (runs/ = first production, to be renamed runs_cap250/), seg 3,4,5 (corner, capsule,
  ellipsoid) invert 6 surface tets during the slide (J ~ -0.9, inv_check.py), they STAY inverted after lift-off
  (uz 830 um with zero force): edge-length (SVK) energy is reflection invariant -> mirrored tets are stress free.
  Inversion happens when the stuck surface is dragged ~300-400 um (Ft ~ 0.85 N < mu Fz).
- Production #2 (prod/): DEPTH_CAP 0.10 mm, per-frame min J / n inverted stored (f_minJ, f_ninv). logs/prod_seg*.log
- prod/ (cap 0.10 mm, solref 1e-5): seg 0,3,4,5 done, no inversion (min J 0.755-0.877); force-weighted contact
  penetration 15-43 um at 97 um travel (ellipsoid worst).  test/sr3e-6, test/sr1e-6 (seg 5): pen_fw 14.9 / 14.8 um,
  identical -> saturated; Fz 0.726 N.  => prod2/ = cap 0.10 mm, solref 3e-6 (launched, logs/prod2_seg*.log).
- render.py drafted (RUNS env selects run dir, EXAG env, default 50).
## Session 3 (2026-10-31)
- prod2 verified complete (8 segs, no NaN, no inversion, iter-cap frac 0); analyze.py run -> prod2/checks.json, profile.png.
- render.py revised (EXAG default 40, shaded surface, closer camera, legend below strip, longer shear arrows);
  backup of the earlier draft: render_v1_backup.py. Test frames in frames_test/.
- Launched full render: RUNS=prod2 ./render_all.sh (90 procs, GPUs 0-3) -> frames/, flex_contact_closeup.mp4, stills.png
- DONE (session 3): frames/ (1040 + title), flex_contact_closeup.mp4 (1920x1080, 30 fps, h264 yuv420p, 1250 fr, 41.67 s),
  stills.png, profile.png, prod2/checks.json, prod2/ringing.json (ringing.py), test/cone/cone.json (cone_check.py:
  contact-frame |ft|/fn max 1.0000, normal tilt up to 74.8 deg -> pad-frame Ft/Fz 3.09 is geometric), README.md written.
- Title card fixed: "(stopped by the indentation cap)" instead of "(time)", plus a line on the unreached 5-20 N loads.
- Nothing left except optional exaggerated MuJoCo render (not done).
