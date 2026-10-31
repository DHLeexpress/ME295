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
