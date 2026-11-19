"""
Force-set sampler: z contact forces on one disk, in exact force and torque balance.

Every contact is treated identically (no "last force closes the balance" step):
  1. contact angles phi_k: uniform on the circle conditioned on every neighboring gap >= gap_min
     (default 53 deg = 2 asin(R / (R + R_n)), the closest spacing of two neighbors of radius R_n = 1.24 R),
     sampled exactly via Dirichlet gaps
  2. disk force scale m: log-uniform on [m_lo, m_hi]                     (flat prior in log force)
  3. normal forces   n_k = m * LogNormal(0, sigma_n)
     tangential      t_k = n_k * tan(a_k),  a_k ~ U(-a_max, a_max)
  4. balance: for fixed phi, sum F = 0 and sum torque = 0 are 3 linear constraints on (n, t);
     (n, t) is replaced by its minimum-norm projection onto that null space
  5. rejected unless every n_k > n_min_frac * m, |t_k| <= mu * n_k, every f_k <= f_max and the
     contact-induced fringe order at the disk center is <= nc_max
Defaults: m in [0.02, 4] N, nc_max = 5, f_max = 6.4 N (~6 diametric-equivalent
fringes, 4 f / (pi R fsigma) at R = 6 mm, fsigma = 225), mu = 1.
Returned per contact: f = sqrt(n^2 + t^2), alpha = atan2(t, n), phi. Forces in N for a disk of
radius R_REF; contact convention as contact_force_vectors.
"""
import math
import numpy as np

from render import center_fringe_order


def contact_force_vectors(f, phi, alpha):
    """Cartesian force on the disk: f cos(alpha) along the inward normal -(cos phi, sin phi),
    f sin(alpha) along the tangent (-sin phi, cos phi)."""
    fx = -f * np.cos(alpha) * np.cos(phi) - f * np.sin(alpha) * np.sin(phi)
    fy = -f * np.cos(alpha) * np.sin(phi) + f * np.sin(alpha) * np.cos(phi)
    return fx, fy


def _random_phi(rng, n, min_spacing):
    """n uniform points on the circle conditioned on all gaps >= min_spacing: the gaps are
    min_spacing + (2 pi - n min_spacing) * Dirichlet(1, ..., 1), with a uniform random offset."""
    gaps = min_spacing + (2 * np.pi - n * min_spacing) * rng.dirichlet(np.ones(n))
    return np.sort((rng.uniform(0, 2 * np.pi) + np.cumsum(gaps)) % (2 * np.pi))


def _project_balanced(n, t, phi):
    """Minimum-norm correction of (n, t) so that sum F = 0 and sum t = 0."""
    C = np.vstack([np.r_[-np.cos(phi), -np.sin(phi)],
                   np.r_[-np.sin(phi), np.cos(phi)],
                   np.r_[np.zeros_like(phi), np.ones_like(phi)]])
    x = np.r_[n, t]
    x = x - C.T @ np.linalg.solve(C @ C.T, C @ x)
    return x[:len(phi)], x[len(phi):]


def sample_force_sets(z, n_sets, rng, m_lo=0.02, m_hi=4.0, sigma_n=0.5, a_max=math.radians(40),
                      mu=1.0, gap_min=math.radians(53), n_min_frac=0.02, f_max=6.4, nc_max=5.0, R=0.006, fsigma=225.0,
                      max_attempts=10**5, return_stats=False):
    """Returns f, phi, alpha arrays of shape [n_sets, z], each row sorted by phi."""
    out = np.zeros((3, n_sets, z))
    tries = 0
    for k in range(n_sets):
        for _ in range(max_attempts):
            tries += 1
            phi = _random_phi(rng, z, gap_min)
            m = math.exp(rng.uniform(math.log(m_lo), math.log(m_hi)))
            n = m * rng.lognormal(0, sigma_n, z)
            t = n * np.tan(rng.uniform(-a_max, a_max, z))
            n, t = _project_balanced(n, t, phi)
            if np.all(n > n_min_frac * m) and np.all(np.abs(t) <= mu * n):
                f, alpha = np.hypot(n, t), np.arctan2(t, n)
                if f.max() <= f_max and center_fringe_order(f[None], alpha[None], phi[None], R, fsigma)[0] <= nc_max:
                    out[:, k] = f, phi, alpha
                    break
        else:
            raise RuntimeError(f'no feasible force set for z={z}')
    return (*out, n_sets / tries) if return_stats else tuple(out)

