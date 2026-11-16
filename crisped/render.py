"""
Batched photoelastic disk renderer: the forward model of CRISPeD.

Superposed point-load (Flamant) stresses from each contact, plus an optional axisymmetric residual stress
    sigma_r = K (R^p - r^p),  sigma_theta = K (R^p - (p+1) r^p),  K = A fsigma / (2 p R^p),
with tension positive (the Flamant term is compressive). The amplitude A is the `halo` argument: A = 0 is a disk
without residual stress, A = +1 the swollen rim of the paper (hoop compression at the rim, half a fringe order
there), A = -1 the opposite sign (a shrunken rim).

Contacts are padded to a common z with f = 0; a zero force contributes zero stress.
"""
import math
import torch

# residual-stress amplitude A of each image condition / forward model
HALO_AMP = {'clean': 0.0, 'halo': 1.0}


def disk_grid(px, rm, device, dtype=torch.float64):
    """Pixel-centre coordinates on a px x px grid spanning [-rm, rm], 'ij' indexing."""
    x = torch.linspace(-rm, rm, px, device=device, dtype=dtype)
    xxi, xxj = torch.meshgrid(x, x, indexing='ij')
    inside = xxi**2 + xxj**2 <= rm**2
    return xxi, xxj, inside


def center_fringe_order(f, alpha, beta, rm, fsigma, device='cpu'):
    """Contact-induced fringe order dsigma/fsigma at the disk centre (no halo), shape [B]."""
    f = torch.as_tensor(f, dtype=torch.float64, device=device)
    alpha = torch.as_tensor(alpha, dtype=torch.float64, device=device)
    beta = torch.as_tensor(beta, dtype=torch.float64, device=device)
    xi = torch.zeros(1, 1, 1, dtype=torch.float64, device=device)
    return _contact_dsigma(f, alpha, beta, xi, xi, rm)[:, 0] / fsigma


def _contact_stress(f, alpha, beta, xi, xj, rm, eps=1e-10):
    """Superposed contact stresses at points (xi, xj) [1,1,P]; returns sxx, syy, sxy [B,P]."""
    beta_adj = (-beta + math.pi / 2)[..., None]
    a = alpha[..., None]
    b = beta_adj + math.pi / 2
    b2 = torch.where(a < 0, b + (math.pi + 2 * a), b - (math.pi - 2 * a))

    x1, y1 = rm * torch.sin(b), rm * torch.cos(b)
    ch0, ch1 = rm * torch.sin(b2) - x1, rm * torch.cos(b2) - y1
    chn = torch.sqrt(ch0**2 + ch1**2 + eps)
    ch0, ch1 = ch0 / chn, ch1 / chn

    r10 = xi - x1
    r11 = -xj - y1
    r1n = torch.sqrt(r10**2 + r11**2 + eps)
    costh1 = torch.clamp((r10 * ch0 + r11 * ch1) / r1n, -1 + eps, 1 - eps)
    signth = torch.where(r11 * ch0 > r10 * ch1, 1.0, -1.0)
    th = signth * torch.acos(costh1) - beta_adj - a
    s1 = -(f[..., None] * 2 / math.pi) / r1n * costh1
    return (s1 * torch.sin(th)**2).sum(1), (s1 * torch.cos(th)**2).sum(1), (0.5 * s1 * torch.sin(2 * th)).sum(1)


def _contact_dsigma(f, alpha, beta, xi, xj, rm):
    sxx, syy, sxy = _contact_stress(f, alpha, beta, xi, xj, rm)
    return torch.sqrt((sxx - syy)**2 + 4 * sxy**2)


def render_batch(f, alpha, beta, rm, px, fsigma, halo=0.0, power=6.0, device='cuda', eps=1e-10,
                 dtype=torch.float64):
    """
    f, alpha, beta : [B, z] tensors (force magnitude [N], force angle, contact angle [rad]).
    Returns [B, px, px] float32 intensity sin^2(pi * dsigma / fsigma), zero outside the disk.
    halo: residual-stress amplitude A (see the module docstring), power: its exponent p.
    Differentiable in f and alpha (used by fit.py with dtype=float32).
    """
    f = torch.as_tensor(f, dtype=dtype, device=device)
    alpha = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta = torch.as_tensor(beta, dtype=dtype, device=device)
    xxi, xxj, inside = disk_grid(px, rm, device, dtype)
    xi = xxi[inside][None, None, :]
    xj = xxj[inside][None, None, :]
    sxx, syy, sxy = _contact_stress(f, alpha, beta, xi, xj, rm, eps)

    if halo:
        R = torch.sqrt(xi[0, 0]**2 + xj[0, 0]**2 + eps)
        K = halo * fsigma / (2 * power * rm**power)
        s_r = K * (rm**power - R**power)
        s_t = K * (rm**power - (power + 1) * R**power)
        T = torch.atan2(xi[0, 0], xj[0, 0])
        c2, s2, sc = torch.cos(T)**2, torch.sin(T)**2, torch.sin(T) * torch.cos(T)
        sxx = sxx + s_r * c2 + s_t * s2
        syy = syy + s_r * s2 + s_t * c2
        sxy = sxy + (s_r - s_t) * sc

    dsig = torch.sqrt((sxx - syy)**2 + 4 * sxy**2)
    inten = torch.nan_to_num(torch.sin(math.pi / fsigma * dsig)**2)

    out = torch.zeros(f.shape[0], px * px, dtype=dtype, device=device)
    out = out.index_copy(1, inside.reshape(-1).nonzero()[:, 0], inten)
    return out.reshape(-1, px, px).float()
