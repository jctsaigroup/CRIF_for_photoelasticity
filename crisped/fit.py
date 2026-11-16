"""
Per-disk contact-force fit, batched over disks, started from an initial guess (CRISPeD or any other).

Per disk (independent; Adam's per-element updates make batching exact):
  synth  = render(|f|, alpha, phi)                      (float32)
  loss   = mean_px (smooth3x3(synth) - target)^2
         + w_torque * (sum_k sin(alpha_k) R |f_k|)^2                              (torque balance, 1e5)
         + w_force * [(sum_k cos(alpha_k - phi_k + pi/2)|f_k|)^2 + (sum_k sin(...)|f_k|)^2]   (force balance, 1)
  Adam(lr) on (f, alpha); a disk stops when its loss improves by < tol for `patience` consecutive
  iterations, or at n_iter. Stopped disks are frozen (parameters restored after each step).
Both the signed parameter f and |f| are returned. SETTINGS holds the loose and tight fits of the paper.
The target is the disk frame on an absolute intensity scale (calibrated_target), not divided by its maximum.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from render import render_batch

FSIGMA = 225.0
# fit settings of the paper: loose (stops early) and tight (run to convergence)
SETTINGS = {'loose': dict(lr=4e-2, n_iter=1000, tol=1e-3, patience=10),
            'tight': dict(lr=4e-2, n_iter=2000, tol=1e-6, patience=100)}


def calibrated_target(frame_u8, gain, offset):
    """Fit target on the absolute intensity scale: blur 3x3 sigma 1, grey level / 255, remove one gain and offset
    calibrated for the whole experiment, mask. A faint disk stays faint: dividing each frame by its maximum would
    scale it up to full brightness and make the fit overestimate weak forces."""
    px = frame_u8.shape[0]
    b = cv2.GaussianBlur(frame_u8, (3, 3), 1).astype(np.float32) / 255.0
    b = np.clip((b - offset) / gain, 0, None)
    yy, xx = np.ogrid[:px, :px]
    return b * (((xx - px // 2) ** 2 + (yy - px // 2) ** 2) <= (px // 2) ** 2)


def intensity_levels(frames_u8):
    """Gain and offset of a set of disk frames from their own black and white levels, no forces needed: per frame the
    0.5 and 99.5 percentiles of the blurred grey levels inside r < 0.93 R (the edge band is excluded); offset = 5th
    percentile of the lows, gain = 95th percentile of the highs minus the offset. A least-squares fit of the frames on
    a model image is not used: any misalignment of the fringes dilutes its slope (gain too low, offset too high)."""
    px = frames_u8[0].shape[0]
    yy, xx = np.ogrid[:px, :px]
    mask = (xx - (px - 1) / 2) ** 2 + (yy - (px - 1) / 2) ** 2 <= (0.93 * px / 2) ** 2
    lv = np.array([np.percentile(cv2.GaussianBlur(f, (3, 3), 1)[mask], [0.5, 99.5]) for f in frames_u8]) / 255.0
    offset = np.percentile(lv[:, 0], 5)
    return dict(gain=float(np.percentile(lv[:, 1], 95) - offset), offset=float(offset))


def _smooth(x):
    k = torch.arange(3, dtype=x.dtype, device=x.device) - 1.0
    g = torch.exp(-k ** 2 / 2)
    g = g / g.sum()
    return F.conv2d(x[:, None], (g[:, None] @ g[None, :])[None, None], padding=1)[:, 0]


def fit_batch(target, f0, a0, phi, rm, halo, lr=4e-2, n_iter=1000, tol=1e-3, patience=10,
              power=6.0, device='cuda', fsigma=FSIGMA, w_torque=1e5, w_force=1.0, snapshots=None):
    """target [B, px, px]; f0, a0, phi [B, z] (same z within a batch).
    halo: residual-stress amplitude of the forward model (render.py), 0 for disks without residual stress.
    w_torque, w_force: weights of the equilibrium penalties (0 switches a term off).
    snapshots: optional iteration counts k; the result then has 'snap' = {k: (f_abs, alpha)} after k Adam steps
    (k = 0 is the start; a disk that stopped early keeps its final values).
    Returns dict of numpy arrays: f (signed), f_abs, alpha, loss, image_loss, n_iter."""
    T = torch.as_tensor(target, dtype=torch.float32, device=device)
    f = torch.tensor(np.asarray(f0), dtype=torch.float32, device=device, requires_grad=True)
    a = torch.tensor(np.asarray(a0), dtype=torch.float32, device=device, requires_grad=True)
    ph = torch.as_tensor(np.asarray(phi), dtype=torch.float32, device=device)
    B, px = T.shape[0], T.shape[-1]
    opt = torch.optim.Adam([f, a], lr=lr)
    active = torch.ones(B, dtype=torch.bool, device=device)
    prev = torch.full((B,), float('inf'), device=device)
    counter = torch.zeros(B, dtype=torch.long, device=device)
    iters = torch.full((B,), n_iter, dtype=torch.long, device=device)
    last_loss = torch.zeros(B, device=device)
    last_img = torch.zeros(B, device=device)
    snap_at, snap = set(snapshots or ()), {}

    def take(k):
        if k in snap_at:
            snap[k] = (f.detach().abs().cpu().numpy(), a.detach().cpu().numpy())

    take(0)
    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        fa = f.abs()
        synth = render_batch(fa, a, ph, rm, px, fsigma, halo=halo, power=power, device=device, dtype=torch.float32)
        img = ((_smooth(synth) - T) ** 2).mean((1, 2))
        ang = a - ph + math.pi / 2
        torque = w_torque * (torch.sin(a) * rm * fa).sum(1) ** 2
        force = w_force * ((torch.cos(ang) * fa).sum(1) ** 2 + (torch.sin(ang) * fa).sum(1) ** 2)
        loss = img + torque + force
        with torch.no_grad():
            last_loss = torch.where(active, loss.detach(), last_loss)
            last_img = torch.where(active, img.detach(), last_img)
        (loss * active).sum().backward()
        f_keep, a_keep = f.detach().clone(), a.detach().clone()
        opt.step()
        with torch.no_grad():
            f[~active] = f_keep[~active]
            a[~active] = a_keep[~active]
            improved = (prev - loss.detach()) >= tol
            counter = torch.where(improved, torch.zeros_like(counter), counter + 1)
            counter = torch.where(torch.isinf(prev), torch.zeros_like(counter), counter)
            prev = torch.where(active, loss.detach(), prev)
            stop = active & (counter >= patience)
            iters[stop] = it + 1
            active &= ~stop
        take(it + 1)
        if not active.any():
            break
    for k in snap_at - set(snap):     # every disk stopped before k: final values
        snap[k] = (f.detach().abs().cpu().numpy(), a.detach().cpu().numpy())
    out = dict(f=f.detach().cpu().numpy(), f_abs=f.detach().abs().cpu().numpy(), alpha=a.detach().cpu().numpy(),
               loss=last_loss.cpu().numpy(), image_loss=last_img.cpu().numpy(), n_iter=iters.cpu().numpy())
    if snapshots:
        out['snap'] = snap
    return out
