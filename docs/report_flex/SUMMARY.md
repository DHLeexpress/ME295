# MuJoCo flex for the PDMS skin, in plain terms

The full report is [flex_study.pdf](flex_study.pdf) (5 pages). The close-up contact video is
[docs/media/flex_contact_closeup.mp4](../media/flex_contact_closeup.mp4). Every number below comes from a run in
[study/](study/) or from the MuJoCo 3.17.0 source. Errors are against exact solutions or a converged FEM, not against
the Winkler cell model.

1. **A MuJoCo bug makes the layer too stiff in compression.** The tetrahedral flex (`dof="full"`) uses twice the
   correct Lamé λ. A confined layer comes out 1.82× too stiff at ν = 0.45; shear is exact. Fix: enter
   `young="971787" poisson="0.409091"` for E = 1 MPa, ν = 0.45 (993552 / 0.480392 for ν = 0.49). The generator
   (`scripts_sim/make_blade_assets.py`, `flex_lame_inputs`) does this. The upstream bug report is drafted, not posted.
2. **Ground truth without MuJoCo.** An exact integral-transform solution of the bonded layer (with a boundary-element
   contact solve) and a converged quadratic finite-element model agree to 0.12 %. Whole-face platen: 8.566 N/µm;
   one-taxel punch: 0.2166 N/µm (ν = 0.45). Winkler is 7–35 % too stiff at ν = 0.45, 1.2–3.7× at ν = 0.49 and up to
   30× at ν = 0.499.
3. **With the fix, MuJoCo's elastic operator converges to the exact answer** from the stiff side, about in proportion
   to the cell size: +2.6 % for the whole face and +36 % for a one-taxel punch at 1.3 mm cells.
4. **Contact softness is the biggest error, and it is where overmeshing hurts.** MuJoCo's contact acts as a spring in
   series with the layer, and its stiffness scales with the vertex mass. Thinner elements mean lighter vertices and a
   softer contact: with more than 2 elements through the 1.5 mm, the total error gets *worse* (−12 % at 2, −36 % at
   6), each element costs 1.3–1.4× more and the solver needs more iterations. In-plane cells below about 2.5 mm add
   cost without gain for whole-pad loads; for one-taxel loads the error falls only about linearly with cell size.
5. **Recommended meshes.**
   - Robot: 6×8×2 vertices (one element per taxel). With a stiff contact and 1000 solver iterations it costs
     21 ms per 0.5 ms step and gives realistic compression; with a softer contact (solref 1 ms) it costs 6 ms but
     the contact, not the PDMS, then supplies most of the compliance (taxel forces stay correct).
   - Fixed rig: 41×57×3 vertices, about 13 % too soft for whole-pad loads (from contact compliance).
   - Indenter at a known position: a mesh refined under it, within 12–21 %.
   - `trilinear`/`quadratic` flexes are exact in elasticity but unusable with contact (50-contact cap, shear fails).
6. **The robot blow-up is a second MuJoCo bug.** The per-island constraint solver leaves out the stiffness of a flex
   pinned to a jointed body. Fix: `<flag island="disable"/>` (or the one-function engine patch in
   `study/robot/fix/`). A stiff flex contact also needs a converged solve (CG 1000 iterations), otherwise transient
   overloads turn the layer inside out (the edge-length energy has no inversion barrier).
6b. **The full librarian task does not work with the flex yet.** It ran without divergence (35 min wall), but the book
   turned 31° in the grasp when it left the shelf. Two contact effects: a box touches the coarse flex at tet corners,
   which pulls the contact force toward the pad centre and halves the torsional grip (the box starts to turn at
   0.58 N m on the 6×8×2 flex vs 1.11 N m on the cells and in the exact solution, at 24 N per pad), and MuJoCo's
   force/torque sensors skip flex contacts (`mj_rnePostConstraint`), so the wrist F/T and its guards do not see the
   pads. The demo therefore stays on the cell model for now. A finer 11×15×2 pad does not close the gap (0.49 N m): the flex's contacts reach only ~30 mm from the pad centre, against 38–40 mm for the cells.
7. **Contact is not defined per taxel.** It exists where surfaces touch; taxels only bin the contact forces. Pressing
   one taxel makes its neighbours bulge slightly *outward* (3.5 % of the indentation at ν = 0.45, 13 % at 0.49, 20 %
   at 0.499), and they carry no force. A "+1, +2, +1" pattern means the outer cells are touched too, or the mesh is
   too coarse (on the old 11×15×3 robot grid a one-taxel push spills +0.22 to +0.25 onto the neighbours).
8. **Still open.** Report the three engine defects upstream (doubled λ, island metric, interpolated-flex force path);
   try MuJoCo's experimental Neo-Hookean option; real PDMS (ν ≈ 0.49–0.4998) doubles the contact error; the layer's
   dynamics are numerically over-damped, so treat the skin as quasi-static.
