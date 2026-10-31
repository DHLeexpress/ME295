# Writes sources.json (one record per source cited in NOTES.md). URLs were checked (HTTP 200) in urls_status.txt.
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
GH = "https://github.com/"
MJ = GH + "google-deepmind/mujoco/"
S = []
def add(id, url, kind, what, settings=None, validation=None, copy_avoid=None, local=None):
    S.append(dict(id=id, url=url, kind=kind, what=what, settings=settings, validation=validation, copy_avoid=copy_avoid, local=local))
add("O-doc", MJ + "blob/3.17.0/doc/XMLreference.rst", "official doc", "flex/flexcomp elasticity attributes; articulated attachments", None, None, "nothing on 3D StVK volumetric normalization", "src/mujoco/doc/XMLreference.rst")
add("O-int", MJ + "blob/3.17.0/doc/computation/index.rst", "official doc", "discrete integrator, metric M+hD+h^2K, limitations", None, None, "use discrete+CG; PGS/noslip error with flex", "src/mujoco/doc/computation/index.rst")
add("O-chg", MJ + "blob/3.17.0/doc/changelog.rst", "official doc", "3.13 #4 discrete; 3.14 #1 ipc; 3.15 #5 SNH, #7 attachments; 3.17 #8 interp shear fix, #10 contact cap", None, None, None, "src/mujoco/doc/changelog.rst")
add("O-mjx", MJ + "blob/3.17.0/doc/mjx.rst", "official doc", "MJX: Flex not supported; MJWarp: VERTCOLLIDE, ELASTICITY", None, None, "no MJX for flex", "src/mujoco/doc/mjx.rst")
add("O-ex", MJ + "tree/3.17.0/model/flex", "official examples", "35 example files tabulated and stepped", "see official_examples_table.txt", "none in files", "only gripper_trilinear has an elastic 3D flex on moving bodies", "official_examples.json")
add("O-test", MJ + "blob/3.17.0/test/engine/engine_passive_test.cc", "official test", "ElasticEnergySolid uses poisson 0 (cannot detect F1)", None, None, None, None)
add("W-readme", GH + "google-deepmind/mujoco_warp/blob/main/README.md", "official", "Flex experimental", None, None, None, "raw/mjwarp/README.md")
add("W-io", GH + "google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/io.py", "official code", "rejects dof=quadratic; noslip+discrete", None, None, None, "raw/mjwarp/mujoco_warp__src_io.py")
add("W-test", GH + "google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/flex_test.py", "official tests", "parity tests vs MuJoCo C only", None, "parity, not physics", "inherits F1", "raw/mjwarp/mujoco_warp__src_flex_test.py")
for n, t in [(1709, "Warp does not cap flex-geom contacts at mjMAXCONPAIR; cloth sinks with cap 50 in C 3.14, rests with 4000"),
             (1723, "3D flex stiffness 21 -> 24 values per element (SNH), fixed by #1724"),
             (1859, "flex contacts attributed to unrelated rigid touch/force sensors (open)")]:
    add(f"W-{n}", GH + f"google-deepmind/mujoco_warp/issues/{n}", "issue", t, local=f"raw/issues_mjwarp/{n}.md")
add("W-1612", GH + "google-deepmind/mujoco_warp/pull/1612", "PR", "Warp tactile sensor fixes (SDF/mesh tactile, not flex)", local="raw/issues_mjwarp/1612.md")
for n, k, t in [(2500, "discussions", "StVK accuracy; dev: linear elements mesh-dependent; damping = value x Lame params; internal=false"),
                (1777, "discussions", "cantilever vs flex off ~8x; dev: p1 coarse poor, use cable plugin"),
                (2652, "discussions", "floppy at 10 ms resets; dev: 1 ms near stability limit (explicit era)"),
                (2641, "discussions", "flex contact forces via mjContact.flex"),
                (2084, "issues", "soft gripper E 11e6 nu .45 diverges; dev: reduce timestep, contact always compliant"),
                (2077, "issues", "attach soft to rigid: child flexcomp + pin"),
                (1922, "issues", "touch_grid on rigid gives one contact; flex contacts fix it but slow"),
                (3364, "issues", "trilinear gripper on rotated parent -> NaN"),
                (3379, "pull", "fix: passive forces assumed world-aligned slides"),
                (3437, "pull", "fix stretch stiffness basis for rotated parents"),
                (3386, "pull", "effective metric in CG"),
                (3439, "pull", "block preconditioner replaces factorization; qacc_smooth iterative, mjWARN_INERTIA at cap"),
                (3617, "pull", "bending attachments"),
                (3618, "pull", "flex elasticity through vertex attachment Jacobians; CG+discrete; PSD approx, no Jacobian derivatives; mass preconditioner"),
                (3620, "pull", "share StVK element calculations"),
                (3621, "pull", "SNH via mjsFlex.elastic3d=1; 'previous 3D StVK volumetric normalization is preserved'"),
                (3632, "pull", "cache flex Hessians"),
                (3638, "pull", "SNH Hessian PSD projection; benchmark incl. 7^3 grid on hinge"),
                (3297, "issues", "filterFlexContacts compaction bug"),
                (3562, "pull", "deterministic tie-breaking in filterFlexContacts")]:
    add(f"M-{n}", MJ + f"{k}/{n}", k, t, local=f"raw/{'discussions' if k=='discussions' else 'issues'}/{n}.md")
add("T1", GH + "badinkajink/hand/blob/main/scripts/contact_bed_candidates.py", "third-party code", "TPU fingertip flex cap (closest analogue)",
    "dim 3 full direct; 1 mm hex columns, 2 x 1.5 mm layers; E 1e7 nu .45 beta 1e-3; radius 0; solref .002 1, solimp .9999 .9999 .001 .5 2, priority 1, internal false; back layer + rim pinned; discrete CG 1 ms",
    "4x4x3 mm block 1 N: 19.0 um vs exact 18.75 at nu=0 (their measurement)", "copy pinning/priority/solimp; avoid nu=0 validation; dropped for cost (1249 us/step)", "raw/repos/badinkajink/hand/scripts/contact_bed_candidates.py")
add("T1b", GH + "badinkajink/hand-archive/blob/main/docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html", "third-party doc", "flex cap results vs other pad models", None, "see T1", None, "raw/repos/badinkajink/hand-archive/docs/experiments/20261007-native_compliance/")
add("T2", GH + "badinkajink/hand-archive/blob/main/scripts/flexcomp_skin_probe.py", "third-party code", "compile/cost probe of flexcomp skins (MuJoCo 3.14)", "dt 1 ms discrete; solref .002 1; solimp .9999", "none (not a study)", "vertex mass makes contacts soft", "raw/repos/badinkajink/hand-archive/scripts/flexcomp_skin_probe.py")
add("T2b", GH + "badinkajink/hand-archive/blob/main/docs/experiments/20261004-codex/data/flexcomp_skin_probe.json", "third-party data", "probe results: 16-102 us/step; error messages", local="raw/repos/badinkajink/hand-archive/docs/experiments/20261004-codex/data/flexcomp_skin_probe.json")
add("T3", GH + "Msornerrrr/in-hand-rotation-mjlab/blob/main/src/in_hand_rotation_mjlab/robots/leap_hand/xmls/left_hand_flexpad.xml", "third-party model", "LEAP flex pads (unused variant)", "dim 3 full grid 2x3x3, spacing 2.5/10/10 mm, E 1e5 nu .45, radius 1.2 mm, solref .006 1, implicitfast", "none", "implicitfast + elasticity errors on >=3.13", "raw/repos/Msornerrrr/...")
add("T4", GH + "Msornerrrr/in-hand-rotation-mjlab/blob/main/src/in_hand_rotation_mjlab/robots/leap_hand/pad_model.py", "third-party code", "shipped pad model: rigid slabs + solimp (.30,.99,.003,.5,2)", None, "calibrated to PaXini readings (their claim)", "baseline to beat", "raw/repos/Msornerrrr/...")
add("T5", GH + "Ju6276/WowSkin-Lab/blob/main/wowskin_lab/core/geometry.py", "third-party code", "magnetic skin as 1-cell trilinear flex on Panda/Robotiq", "dim 3 trilinear direct; E 3e4 nu .35 damping 1e-5; radius 0.05 mm; solref 2 ms; 4 bottom nodes pinned; discrete CG 500 us (robot)", "none", "fallback for robot-mounted skin", "raw/repos/Ju6276/WowSkin-Lab/wowskin_lab/core/geometry.py")
add("T5b", GH + "Ju6276/WowSkin-Lab/blob/main/docs/MODEL_CALIBRATION_ZH.md", "third-party doc", "full-node StVK/SNH inverted ~3.41 s; costs; uncalibrated params", None, "none", "SNH has no anti-inversion barrier", "raw/repos/Ju6276/WowSkin-Lab/docs/MODEL_CALIBRATION_ZH.md")
add("T5c", GH + "Ju6276/WowSkin-Lab/blob/main/TODO.md", "third-party doc", "todo: flex gel needs material/boundary calibration", local="raw/repos/Ju6276/WowSkin-Lab/TODO.md")
add("T6", GH + "pulipakaa24/MetalSim/blob/main/docs/research/deformables_2026-09-25.md", "third-party doc", "flex measurements: contact cap, locking sweep, explicit limits", "rod E 1e5 nu .4; cloth E 1e6 nu .45", "cantilever EI_eff/EI 14.0 / 7.0 (their measurement)", "copy mesh sweeps", "raw/repos/pulipakaa24/MetalSim/docs/research/deformables_2026-09-25.md")
add("T7", GH + "TX-Leo/HumanEgo/blob/main/simulation/docs/upgrade/physics.md", "third-party doc", "contact compliance hides modulus (6.8 vs 20 kPa)", "trilinear 4x2x2, E 2e4 nu .25 damping .03, discrete 2 ms", "compression bench (their measurement)", "report penetration", "raw/repos/TX-Leo/HumanEgo/simulation/docs/upgrade/physics.md")
add("T7b", GH + "TX-Leo/HumanEgo/blob/main/simulation/docs/physics_params.md", "third-party doc", "parameter table", local="raw/repos/TX-Leo/HumanEgo/simulation/docs/physics_params.md")
add("T8", GH + "cps-test-lab/roqsim/blob/main/roqsim/src/roqsim/flex_modes.py", "third-party code", "measured discrete-integrator rules: zeta=(damping+h)omega/2, omega*h<=0.3", None, "their tests on MuJoCo 3.14", "compute omega*h", "raw/repos/cps-test-lab/roqsim/roqsim/src/roqsim/flex_modes.py")
add("T8b", GH + "cps-test-lab/roqsim/blob/main/roqsim/src/roqsim/flex.py", "third-party code", "flex rules (discrete required)", local="raw/repos/cps-test-lab/roqsim/roqsim/src/roqsim/flex.py")
add("T9", GH + "paulotto/hybrid-jaw-exo-sim/blob/main/assets/mjcf/exoskeleton/jaw/biorob_2026/jaw_exo_model_rigid_skin_sep_compiled_3_9_MPa_0_45.xml", "third-party model", "soft face skin (2D shell) under jaw exo", "dim 2 stretch, E 3.9e6 nu .45 thickness 2 mm damping 1e-5; solref .002 1.5; 30 us implicit", "none", "not applicable to 3D layer", None)
add("T9b", GH + "paulotto/hybrid-jaw-exo-sim/blob/main/src/exoforge_evaluation/scripts/material_property_converter.py", "third-party code", "E/nu <-> edge-constraint solref converter")
add("T10", GH + "dap-i/FOCUS-Hand/blob/main/Mujoco/wonik_allegro/left_hand_soft.xml", "third-party model", "soft trilinear cube grasped by Allegro", "9x9x9, E 6e3 nu .3 damping .1, radius 1 mm margin 3 mm, solref .03 2, solimp .8 .95 .005, 0.5 ms implicitfast Newton")
add("T11", GH + "Albusgive/mujoco_learning/blob/main/extend/touch/touch_flex.xml", "third-party demo", "flex touch pad, reads per-vertex contact forces", "dim 2, E 3e5 nu 0 elastic2d stretch, 0.5 ms implicitfast")
add("T12", GH + "EduardGilM/Thekenyos/blob/main/scripts/kiwi_compression.py", "third-party code", "kiwi compression bench", "dim 3 full 7^3, E 1.57e6 nu .4, 10 us implicitfast Newton")
add("T13a", GH + "dblanm/benchmarking_cloth/blob/master/bcm/assets/mujoco3_cloth.xml", "third-party model", "cloth sim-to-real benchmark (arXiv 2310.09543)", "shell plugin E 85242 nu .403 thickness .5 mm, 10 ms implicitfast CG")
add("T13b", GH + "google-deepmind/aloha_sim/blob/main/aloha_sim/assets/mujoco/Towel/original.xml", "official-adjacent model", "towel", "dim 2, E 60 nu .1 bend, edge equality solref .002 5")
add("T14", GH + "dfki-ric/deformable_gym/blob/main/deformable_gym/assets/objects/mjcf/insole_fixed.xml", "third-party model", "gmsh insole", "E 1e4 nu .1 + edge equality (now a compile error)")
for a, t in [("2412.03191", "soft adaptive feet (abstract only)"), ("2502.18749", "soft head + articulated jaw (abstract + HTML grep)"),
             ("2606.22397", "TDCRs natively in MuJoCo, validated vs Cosserat rod (abstract only)"), ("2501.08077", "HydroelasticTouch (abstract only)"),
             ("2310.09543", "cloth sim-to-real benchmark (abstract only)")]:
    add("P-" + a, "https://arxiv.org/abs/" + a, "paper", t)
json.dump(S, open(os.path.join(HERE, "sources.json"), "w"), indent=1)
print(len(S), "sources")
