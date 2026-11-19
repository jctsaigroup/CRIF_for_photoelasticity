"""
Smoke test: random balanced force sets -> rendered and modified disks -> CRIF guess -> fit.

    python tests/test_smoke.py [--weights-dir weights] [--n 8]

Without weights only the forward model, the image modifications, the crops and the fit (started from the true forces
perturbed by 20%) are tested. With the released weights the guess of each network is checked on disks of its own
condition. Runs in a few minutes on a CPU with the default 8 disks per condition.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'crif'))
from forcesets import sample_force_sets
from generate import CFG, apply_modifications
from infer import guess, load_model, refine
from render import HALO_AMP, render_batch


def disks(n, cond, rng):
    """n disks with 3 contacts: true f, alpha, phi [n, 3] and modified frames (uint8, px_train)."""
    s = dict(CFG['sampler'])
    s['a_max'], s['gap_min'] = np.radians(s.pop('a_max_deg')), np.radians(s.pop('gap_min_deg'))
    f, phi, alpha = sample_force_sets(3, n, rng, R=CFG['R_ref'], fsigma=CFG['fsigma'], **s)
    ss = CFG['supersample']
    hi = render_batch(f, alpha, phi, CFG['R_ref'], CFG['px_train'] * ss, CFG['fsigma'], halo=HALO_AMP[cond],
                      power=CFG['halo_power'], device='cpu')
    img = torch.nn.functional.avg_pool2d(hi[:, None], ss)[:, 0].numpy()
    mods = dict(blur=1.0, gain=0.8, offset=0.025, noise=0.01)
    frames = [apply_modifications(im, dict(mods, noise_seed=int(rng.integers(2**31)))) for im in img]
    return f, alpha, phi, frames, mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights-dir', default=os.path.join(os.path.dirname(__file__), '..', 'weights'))
    ap.add_argument('--n', type=int, default=8, help='disks per condition')
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    ok = True
    for cond in ('clean', 'halo'):
        f, alpha, phi, frames, mods = disks(a.n, cond, rng)
        start = f * rng.uniform(0.8, 1.2, f.shape)
        ff, _ = refine(frames, list(phi), list(start), list(alpha), CFG['R_ref'], CFG['fsigma'], HALO_AMP[cond],
                       mods['gain'], mods['offset'], 'tight')
        e_fit = np.median(np.abs(np.stack(ff) / f - 1))
        line = f'{cond}: fit from perturbed truth {100 * e_fit:.1f}%'
        ok &= e_fit < 0.05
        w = os.path.join(a.weights_dir, f'crif_{cond}.pt')
        if os.path.exists(w):
            fg, ag = guess(load_model(w), frames, list(phi), CFG['R_ref'], CFG['fsigma'])
            e_g = np.median(np.abs(np.stack(fg) / f - 1))
            e_a = np.degrees(np.median(np.abs(np.stack(ag) - alpha)))
            line += f', guess {100 * e_g:.1f}% / {e_a:.2f} deg'
            ok &= e_g < 0.05
        else:
            line += ' (no weights, guess not tested)'
        print(line, flush=True)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
