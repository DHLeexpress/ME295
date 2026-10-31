# DRAFT (not posted): dof="full" 3D flex elasticity uses twice the Lame lambda

**Title:** Flex StVK elasticity (dim=3, dof="full") uses 2*lambda: nonzero Poisson ratio gives the wrong bulk response

**Version:** MuJoCo 3.17.0 (tag 3.17.0), Python bindings, Linux.

**Description.** For a 3D flex with `<elasticity young=E poisson=nu>`, the nodal forces under an exact homogeneous
strain match a Saint Venant-Kirchhoff material with Lame parameters (2*lambda, mu), not (lambda, mu).
- Simple shear is exact.
- Uniaxial strain is (2 lambda + 2 mu)/(lambda + 2 mu) too stiff: 1.8182 at nu = 0.45.
- Bulk is (2 lambda + 2 mu/3)/(lambda + 2 mu/3) too stiff: 1.9310 at nu = 0.45.

The error is exact for every nu > 0, every mesh, and also at finite strain. `ElasticEnergySolid`
(test/engine/engine_passive_test.cc) uses poisson = 0 and cannot detect it. The experimental SNH path (elastic3d = 1)
and the trilinear/quadratic flexes are correct.

**Repro (17 lines):**
```python
import mujoco, numpy as np
E, nu, eps = 1e6, 0.45, 1e-6           # uniaxial strain e_xx = eps on one free tetrahedron
xml = f"""<mujoco><worldbody><flexcomp name="t" type="direct" dim="3" radius="0" mass="1"
  point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
  <elasticity young="{E}" poisson="{nu}"/></flexcomp></worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
d.qpos[:] = (X @ np.diag([eps, 0, 0])).ravel()   # u = eps*x (vertex bodies: 3 slides each)
mujoco.mj_forward(m, d)
f_mj = d.qfrc_spring.reshape(4, 3)
lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
G = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])   # grad N_a
sig = np.diag([lam + 2*mu, lam, lam]) * eps                     # linear elasticity
f_ex = -(1/6) * G @ sig                                          # f_a = -V sigma grad N_a
print("x-force on vertex 1: mujoco %.6g  exact %.6g  ratio %.4f" % (f_mj[1, 0], f_ex[1, 0], f_mj[1, 0]/f_ex[1, 0]))
print("y-force on vertex 2: mujoco %.6g  exact %.6g  ratio %.4f" % (f_mj[2, 1], f_ex[2, 1], f_mj[2, 1]/f_ex[2, 1]))
print("expected ratio with lambda doubled: %.4f (x), 2.0000 (y)" % ((2*lam + 2*mu)/(lam + 2*mu)))
```
**Expected:** both ratios 1.0000. **Actual (3.17.0):**
```
x-force on vertex 1: mujoco -1.14943  exact -0.632184  ratio 1.8182
y-force on vertex 2: mujoco -1.03448  exact -0.517241  ratio 2.0000
```
(The y-force is pure lambda*eps, so its ratio is exactly the lambda factor.)

**Cause.**
- `src/user/user_flexcomp.cc`: `ComputeStiffness` sets `la = E nu/((1+nu)(1-2nu)) |V|/4 * thickness` (thickness
  defaults to 4, so la = lambda V). `MetricTensor` then forms `k_ef = mu tr(B_e B_f) + la tr(B_e) tr(B_f)`.
- At run time the energy is `s' k s / 4` (src/engine/engine_core_util.h, `mj_stretchElasticity`), with s the
  squared-edge-length changes and E = (1/2) sum_e s_e B_e.
- So W = V [mu tr(E^2) + lambda (tr E)^2], whereas StVK is W = V [mu tr(E^2) + (lambda/2) (tr E)^2].
- `ComputeSNH` in the same file gets the lambda/2 right, via (lambda + 2mu)/2 (J-1)^2 together with the -mu trace term.

**Suggested fix (one line, in `ComputeStiffness`, not in `MetricTensor`, which SNH reuses with la = -mu):**
```cpp
double la = E * nu / ((1 + nu) * (1 - 2 * nu)) * std::abs(volume) / 8 * thickness;   // was / 4
```
- Equivalent user-side workaround: poisson_in = nu/(2(1-nu)), young_in = E (1 + poisson_in)/(1 + nu). This keeps mu
  and passes lambda/2; it restores all patch ratios to 1.000000.
- Check whether the dim=2 path (`ComputeStiffness<Stencil2D>`, same `la`) should change with it. Its membrane energy is
  also 1/4 of a plane-stress membrane of the documented `thickness`, as encoded in `ElasticEnergyMembrane`; that is a
  separate issue.
- A regression test with poisson > 0 (as above) should be added.
