"""Linear MPC on the TCP position (velocity input), solved with OSQP (Stellato et al. 2020) at 10 Hz on a 1.5 s
horizon, re-planned on every new grasp estimate (receding horizon; the visual-servoing MPC idea of Allibert et al.
2010 in its simplest position-based form):

    min  sum_k |p_k - g|_Q^2 + |u_k|_R^2 + |u_k - u_{k-1}|_S^2 + rho s_k^2 + rho1 s_k
    s.t. p_{k+1} = p_k + dt u_k,  |u_k| <= v_max (per axis),  |u_k - u_{k-1}| <= a_max dt,
         z_k >= z_g                              (never below the grasp height)
         z_k - z_g >= alpha |e . (p_k - g)| - s_k  for e = jaw axis and its normal   (approach funnel, soft)

The funnel keeps the open jaws above the object until they are aligned with it, in one smooth motion (no separate
align / descend stages). Orientation as in every controller: P on the rotation error."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import TopDownGraspController


class MPCGrasp(TopDownGraspController):
    name = "MPC (OSQP, funnel)"
    closed_loop = True

    def __init__(self, *a, N=15, dt=0.1, v_max=(0.12, 0.12, 0.06), a_max=0.6, alpha=1.5, q=(60.0, 60.0, 30.0),
                 qN=10.0, r=0.2, s=4.0, rho=2e4, solve_every=5, **kw):
        super().__init__(*a, **kw)
        self.N, self.dt, self.v_max, self.a_max, self.alpha = N, dt, np.asarray(v_max, float), a_max, alpha
        self.Q, self.qN, self.r, self.s, self.rho = np.asarray(q, float), qN, r, s, rho
        self.solve_every = solve_every
        self.solves = 0
        self.fails = 0

    def reset_approach(self, obs):
        self.u_prev = np.zeros(3)
        self.u_hold = np.zeros(3)
        self.tick = 0

    def solve(self, p0, goal):
        import osqp
        N, dt = self.N, self.dt
        nx, nu = 3 * N, 3 * N
        nv = nx + nu + N
        g = goal.position
        e1 = goal.T_tcp[:3, 0].copy()
        e1[2] = 0
        e1 /= np.linalg.norm(e1)
        e2 = np.array([-e1[1], e1[0], 0.0])
        # cost
        Pd = np.zeros(nv)
        qv = np.zeros(nv)
        for k in range(N):
            w = self.Q * (self.qN if k == N - 1 else 1.0)
            Pd[3 * k:3 * k + 3] = 2 * w
            qv[3 * k:3 * k + 3] = -2 * w * g
        P = sp.diags(Pd).tolil()
        for k in range(N):         # u_k^T R u_k + (u_k - u_{k-1})^T S (...)
            for j in range(3):
                i = nx + 3 * k + j
                P[i, i] += 2 * self.r + 2 * self.s + (2 * self.s if k < N - 1 else 0.0)
                if k < N - 1:
                    P[i, i + 3] += -2 * self.s
                    P[i + 3, i] += -2 * self.s
        for j in range(3):
            qv[nx + j] += -2 * self.s * self.u_prev[j]
        for k in range(N):
            P[nx + nu + k, nx + nu + k] = 2 * self.rho
            qv[nx + nu + k] = self.rho
        P = sp.triu(P.tocsc(), format="csc")
        rows, l, u = [], [], []
        A = sp.lil_matrix((3 * N + 3 * N + 3 * N + 4 * N + N + N, nv))
        r = 0
        for k in range(N):          # dynamics
            for j in range(3):
                A[r, 3 * k + j] = 1.0
                A[r, nx + 3 * k + j] = -dt
                if k > 0:
                    A[r, 3 * (k - 1) + j] = -1.0
                    l.append(0.0)
                    u.append(0.0)
                else:
                    l.append(p0[j])
                    u.append(p0[j])
                r += 1
        for k in range(N):          # speed
            for j in range(3):
                A[r, nx + 3 * k + j] = 1.0
                l.append(-self.v_max[j])
                u.append(self.v_max[j])
                r += 1
        da = self.a_max * dt
        for k in range(N):          # acceleration
            for j in range(3):
                A[r, nx + 3 * k + j] = 1.0
                if k > 0:
                    A[r, nx + 3 * (k - 1) + j] = -1.0
                    l.append(-da)
                    u.append(da)
                else:
                    l.append(self.u_prev[j] - da)
                    u.append(self.u_prev[j] + da)
                r += 1
        for k in range(N):          # funnel: z_k - alpha e.(p_k - g) + s_k >= z_g
            for e in (e1, -e1, e2, -e2):
                A[r, 3 * k + 0] = -self.alpha * e[0]
                A[r, 3 * k + 1] = -self.alpha * e[1]
                A[r, 3 * k + 2] = 1.0
                A[r, nx + nu + k] = 1.0
                l.append(g[2] - self.alpha * (e[0] * g[0] + e[1] * g[1]))
                u.append(np.inf)
                r += 1
        for k in range(N):          # z_k >= z_g
            A[r, 3 * k + 2] = 1.0
            l.append(g[2])
            u.append(np.inf)
            r += 1
        for k in range(N):          # s_k >= 0
            A[r, nx + nu + k] = 1.0
            l.append(0.0)
            u.append(np.inf)
            r += 1
        prob = osqp.OSQP()
        prob.setup(P, qv, A.tocsc(), np.array(l), np.array(u), verbose=False, polishing=False, max_iter=4000,
                   eps_abs=1e-5, eps_rel=1e-5)
        res = prob.solve()
        self.solves += 1
        if res.x is None or res.info.status_val not in (1, 2):
            self.fails += 1
            return None
        return res.x[nx:nx + 3]

    def approach(self, obs, goal):
        p = obs.tcp_pose[:3]
        if self.tick % self.solve_every == 0:
            u0 = self.solve(p, goal)
            self.u_hold = u0 if u0 is not None else np.clip(1.0 * (goal.T_pre[:3, 3] - p), -self.v_max, self.v_max)
            self.u_prev = self.u_hold.copy()
        self.tick += 1
        e = goal.position - p
        return self.u_hold.copy(), bool(np.linalg.norm(e) < self.reach_tol)
