"""Small-alpha check of kappa_s against the exact asymptotics (Argatov et al. arXiv:1203.0918 eqs. 47, 48).
Output: lit_asymptotic_check.out"""
import numpy as np, gt_transform as gt, gt_contact as gc
from scipy.integrate import quad
R, T = 10e-3, 1.5e-3
out = []
for nu in (0.45, 0.5):
    k = 3 - 4 * nu
    Lf = lambda l: (2 * k * np.sinh(2 * l) - 4 * l) / (2 * k * np.cosh(2 * l) + 1 + k * k + 4 * l * l)
    a0 = quad(lambda l: 1 - Lf(l), 0, 60, limit=500, epsabs=1e-13)[0]
    q = 2 * a0 / np.pi
    L = gt.Layer(nu=nu, t=T)
    for al in (0.0025, 0.005, 0.01, 0.02, 0.04):
        d, F, *_ = gc.sphere_fredholm(L, R, al * T, 32)
        ks = F / gt.hertz(L, R, d)[0]
        w = np.sqrt(d * R) / T
        c_alpha = (ks - 1 - q * al) / al ** 2
        c_w = (ks - 1 - q * w) / w ** 2
        out.append(f'nu={nu} alpha={al}: kappa_s={ks:.8f}  (k-1-q a)/a^2={c_alpha:.4f} [eq48: 2a0^2/pi^2={2*a0**2/np.pi**2:.4f}, '
                   f'derived 10a0^2/(3pi^2)={10*a0**2/(3*np.pi**2):.4f}]  (k-1-q w)/w^2={c_w:.4f} [eq47: 14a0^2/(3pi^2)={14*a0**2/(3*np.pi**2):.4f}]')
open('lit_asymptotic_check.out', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
