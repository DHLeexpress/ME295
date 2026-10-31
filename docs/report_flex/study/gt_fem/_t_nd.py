# quick check of the ORIGINAL code: Kronecker vs elementwise, NDCholesky vs SuperLU
import sys, time, numpy as np
sys.path.insert(0, '.')
import fem_layer_orig as fl
import scipy.sparse.linalg as spla
rng = np.random.default_rng(0)
g = fl.Grid3D(np.array([0, 0.7, 1.5, 3.0]), np.array([0, 1.0, 1.6, 2.0, 3.1]), np.array([0, 0.5, 1.5]))
for form in ("full", "mixed"):
    for nu in (0.3, 0.45, 0.499):
        K1 = fl.stiffness(g, 1.0, nu, form); K2 = fl.stiffness_elementwise(g, 1.0, nu, form)
        print(form, nu, "max|Kkron-Kelem| / max|K| =", abs(K1 - K2).max() / abs(K2).max())
# ND Cholesky vs splu on a moderate C1-like problem
bx = fl.graded_breaks(0, 25.5, [5.1, 15.3], [25.5], 0.3, 3.0, 1.4)
by = fl.graded_breaks(0, 35.5, [5.0714, 15.214, 25.357], [35.5], 0.3, 3.0, 1.4)
bz = np.array([0, 0.3, 0.75, 1.2, 1.5])
g = fl.Grid3D(bx, by, bz)
K = fl.stiffness(g, 1.0, 0.45, "full")
ia, ib, ic = [a.ravel() for a in np.meshgrid(np.arange(g.nx), np.arange(g.ny), np.arange(g.nz), indexing='ij')]
A = g.node(ia, ib, ic)
fixed = np.concatenate([3*A[ic == g.nz-1][:, None] + np.arange(3), 3*A[ic == 0][:, None] + np.arange(3)]).ravel()
vals = np.zeros((g.ndof,)); vals[3*A[ic == 0] + 2] = 1.0
sx = 3*A[(ia == 0)]; sy = 3*A[ib == 0] + 1
fixed = np.unique(np.concatenate([fixed, sx, sy]))
v = vals[fixed]
for method in ("nd", "superlu"):
    t0 = time.time()
    U, R, info = fl.solve_dirichlet(K, fixed, v, grid=g, method=method)
    print(method, g.ndof, "Kn quarter*4 [N/mm] =", 4*R[3*A[ic == 0] + 2].sum(), info, "%.1fs" % (time.time()-t0))
