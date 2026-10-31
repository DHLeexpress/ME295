"""Symbolic derivation / verification of the bonded-layer surface compliance (sympy).

Layer 0 <= z <= t, z points INTO the layer, free surface z=0, rigid bonded base z=t.
Fourier mode exp(i k x) (k along x after rotation).  Papkovich-Neuber (P-SV part):
   2 mu u = grad(phi) + grad(z psi) - 4(1-nu) psi e_z ,   lap(phi) = lap(psi) = 0
SH part: u_y = U_y(z) exp(ikx), mu(U_y'' - k^2 U_y) = 0.
Surface tractions applied to the body at z=0: (q_x, q_y, p), p = compressive pressure:
   sigma_xz(0) = -q_x, sigma_yz(0) = -q_y, sigma_zz(0) = -p.
Checks performed here:
  1. the PN representation satisfies Navier's equations identically;
  2. solve the 4x4 boundary-value system symbolically and compare with the closed forms
     C_zz, C_L, C_c (and base tractions) written in gt_transform.py;
  3. limits kt->0 and kt->oo.
"""
import sympy as sp

x, z, k, t, mu, nu = sp.symbols('x z k t mu nu', positive=True)
A, B, C, D, p, q = sp.symbols('A B C D p q')
lam = 2*mu*nu/(1-2*nu)
ex = sp.exp(sp.I*k*x)
phi = (A*sp.exp(-k*z) + B*sp.exp(k*z))*ex
psi = (C*sp.exp(-k*z) + D*sp.exp(k*z))*ex
ux = (sp.diff(phi, x) + sp.diff(z*psi, x))/(2*mu)
uz = (sp.diff(phi, z) + sp.diff(z*psi, z) - 4*(1-nu)*psi)/(2*mu)
div = sp.diff(ux, x) + sp.diff(uz, z)
lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, z, 2)
nav_x = sp.simplify(mu*lap(ux) + (lam+mu)*sp.diff(div, x))
nav_z = sp.simplify(mu*lap(uz) + (lam+mu)*sp.diff(div, z))
print('1. Navier residuals (must be 0):', nav_x, nav_z)

szz = lam*div + 2*mu*sp.diff(uz, z)
sxz = mu*(sp.diff(ux, z) + sp.diff(uz, x))
eqs = [sp.simplify((szz/ex).subs(z, 0) + p),
       sp.simplify((sxz/ex).subs(z, 0) + q),
       sp.simplify((ux/ex).subs(z, t)),
       sp.simplify((uz/ex).subs(z, t))]
sol = sp.solve(eqs, [A, B, C, D], dict=True)[0]
Ux0 = sp.simplify((ux/ex).subs(z, 0).subs(sol))
Uz0 = sp.simplify((uz/ex).subs(z, 0).subs(sol))
u = k*t
kap = 3 - 4*nu
Dp = 2*kap*sp.cosh(2*u) + 1 + kap**2 + 4*u**2
Czz = (1-nu)/(mu*k)*(2*kap*sp.sinh(2*u) - 4*u)/Dp
CL = (1-nu)/(mu*k)*(2*kap*sp.sinh(2*u) + 4*u)/Dp
Cc = (kap*(kap-1)*(sp.cosh(2*u)-1) - 4*u**2)/(2*mu*k*Dp)
# expected: Ux0 = CL q + i Cc p ; Uz0 = -i Cc q + Czz p
chk = lambda e: sp.simplify(sp.expand((e).rewrite(sp.exp)))
r1 = chk(sp.diff(Uz0, p) - Czz)
r2 = chk(sp.diff(Ux0, q) - CL)
r3 = chk(sp.diff(Ux0, p) - sp.I*Cc)
r4 = chk(sp.diff(Uz0, q) + sp.I*Cc)
print('2. closed-form residuals Czz, CL, i*Cc (Ux/p), -i*Cc (Uz/q):', r1, r2, r3, r4)

# base tractions (force per area exerted BY the layer ON the base) = -(sigma_xz, sigma_zz) at z=t
pb = sp.simplify(-(szz/ex).subs(z, t).subs(sol))
tb = sp.simplify(-(sxz/ex).subs(z, t).subs(sol))
pb_cf = (kap+1)/Dp*(p*((1+kap)*sp.cosh(u) + 2*u*sp.sinh(u)) - sp.I*q*((kap-1)*sp.sinh(u) + 2*u*sp.cosh(u)))
tb_cf = (kap+1)/Dp*(-sp.I*p*((1-kap)*sp.sinh(u) + 2*u*sp.cosh(u)) + q*((1+kap)*sp.cosh(u) - 2*u*sp.sinh(u)))
print('   base-traction residuals:', chk(pb - pb_cf), chk(tb - tb_cf))

# SH part
E1, E2, qy = sp.symbols('E1 E2 qy')
Uy = E1*sp.exp(-k*z) + E2*sp.exp(k*z)
solsh = sp.solve([mu*sp.diff(Uy, z).subs(z, 0) + qy, Uy.subs(z, t)], [E1, E2], dict=True)[0]
CT = sp.simplify(Uy.subs(z, 0).subs(solsh)/qy)
print('   SH: C_T =', sp.simplify(CT), ' residual vs tanh(kt)/(mu k):', chk(CT - sp.tanh(u)/(mu*k)))
tby = sp.simplify(-(mu*sp.diff(Uy, z)).subs(z, t).subs(solsh))
print('   SH base traction / q_y =', sp.simplify(tby/qy), ' residual vs 1/cosh:', chk(tby/qy - 1/sp.cosh(u)))

# 3. limits
print('3. limits (t->0 series, leading terms):')
print('   Czz ~', sp.simplify(sp.series(Czz, t, 0, 2).removeO()), '  (expect t/M = t(1+nu)(1-2nu)/(E(1-nu)) = t(1-2nu)/(2 mu (1-nu)))')
print('   CL  ~', sp.simplify(sp.series(CL, t, 0, 2).removeO()), '  (expect t/mu)')
print('   Cc  ~', sp.simplify(sp.series(Cc, t, 0, 3).removeO()))
print('   Czz(nu=1/2) ~', sp.simplify(sp.series(Czz.subs(nu, sp.Rational(1, 2)), t, 0, 4).removeO()), ' (expect k^2 t^3/(3 mu))')
print('   t->oo: Czz*k ->', sp.limit(Czz*k, t, sp.oo), ' CL*k ->', sp.limit(CL*k, t, sp.oo), ' Cc*k ->', sp.limit(Cc*k, t, sp.oo))
print('          expect (1-nu)/mu, (1-nu)/mu, (1-2nu)/(2mu)')
# corrections C - C_halfspace in non-cancelling form (used for numerics)
dzz = -(1-nu)/(mu*k)*(2*kap*sp.exp(-2*u) + (1+2*u)**2 + kap**2)/Dp
dL = -(1-nu)/(mu*k)*(2*kap*sp.exp(-2*u) + (1-2*u)**2 + kap**2)/Dp
dc = -((kap-1)*((kap+1)**2 + 4*u**2) + 8*u**2)/(4*mu*k*Dp)
print('   correction-form residuals:', chk(Czz - (1-nu)/(mu*k) - dzz), chk(CL - (1-nu)/(mu*k) - dL),
      chk(Cc - (1-2*nu)/(2*mu*k) - dc))
