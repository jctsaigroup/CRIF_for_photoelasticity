"""
CRISPeD on experimental disk images: initial guess of the contact forces, optionally refined by the fit.

    python infer.py --weights weights/crisped_halo.pt --image disk.png --phis 0.12 1.75 3.30 \
                    --radius 0.0075 --fsigma 225 [--fit --gain 0.8 --offset 0.05]

Inputs per disk:
  frame   square grayscale uint8 image with the disk filling it (any size: it is resized to the training diameter,
          px_train = 100, with area interpolation before the contact crops are cut)
  phis    contact angles in rad, measured in the image as atan2(row - cy, col - cx)
  radius  physical disk radius R in m, and the photoelastic constant fsigma of the material in N/m
Outputs per contact: force magnitude f in N and angle alpha in rad (alpha = atan2(tangential, normal), normal pointing
into the disk; forcesets.contact_force_vectors gives the Cartesian force on the disk).
The fit needs the frames on an absolute intensity scale: gain and offset of the camera, e.g. from
fit.intensity_levels over many frames of the experiment.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import TargetTransform, to_network_input
from fit import SETTINGS, calibrated_target, fit_batch
from generate import CFG, contact_crops
from model import build_model


def load_model(path, device='cpu'):
    """Released weights (see export_weights.py) -> dict(model, tf, meta)."""
    ck = torch.load(path, map_location=device)
    meta = ck['meta']
    model = build_model(False, meta['dropout'], meta['hidden']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return dict(model=model, tf=TargetTransform(**ck['target_transform']), meta=meta, device=device)


def to_train_size(frame_u8, px=CFG['px_train']):
    if frame_u8.shape[0] == px:
        return frame_u8
    return cv2.resize(frame_u8, (px, px), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def guess(net, frames, phis, radius, fsigma, batch=256):
    """frames: list of uint8 disk images, phis: list of contact-angle arrays (one per disk).
    Returns lists (one array per disk) of f [N] and alpha [rad]."""
    crops, owner = [], []
    for d, (fr, ph) in enumerate(zip(frames, phis)):
        c = contact_crops(to_train_size(fr), np.asarray(ph))
        crops += c
        owner += [d] * len(c)
    X = torch.from_numpy(np.stack(crops))
    m = net['meta']
    y = np.concatenate([net['model'](to_network_input(X[s:s + batch].to(net['device']), m['input_size'])).cpu().numpy()
                        for s in range(0, len(X), batch)])
    f_over_R, alpha = net['tf'].decode(y)
    f = radius * f_over_R * fsigma / m['fsigma_train']       # the network is trained on f/R at fsigma_train
    owner = np.array(owner)
    return [f[owner == d] for d in range(len(frames))], [alpha[owner == d] for d in range(len(frames))]


def refine(frames, phis, f0, a0, radius, fsigma, halo, gain, offset, setting='tight', device='cpu'):
    """Fit started from (f0, a0) with the forward model of residual-stress amplitude halo (0, or +1 for the
    swollen rim). Disks are batched by contact number. Returns lists of f [N] and alpha [rad] per disk."""
    f_out, a_out = [None] * len(frames), [None] * len(frames)
    by_z = {}
    for d, ph in enumerate(phis):
        by_z.setdefault(len(ph), []).append(d)
    for z, idx in by_z.items():
        T = np.stack([calibrated_target(to_train_size(frames[d]), gain, offset) for d in idx])
        res = fit_batch(T, np.stack([f0[d] for d in idx]), np.stack([a0[d] for d in idx]),
                        np.stack([np.asarray(phis[d]) for d in idx]), radius, halo, fsigma=fsigma, device=device,
                        **SETTINGS[setting])
        for k, d in enumerate(idx):
            f_out[d], a_out[d] = res['f_abs'][k], res['alpha'][k]
    return f_out, a_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--image', required=True, help='grayscale disk image, disk filling the square frame')
    ap.add_argument('--phis', type=float, nargs='+', required=True, help='contact angles (rad)')
    ap.add_argument('--radius', type=float, required=True, help='disk radius (m)')
    ap.add_argument('--fsigma', type=float, required=True, help='photoelastic constant (N/m)')
    ap.add_argument('--fit', action='store_true', help='refine the guess with the fit')
    ap.add_argument('--setting', default='tight', choices=list(SETTINGS))
    ap.add_argument('--gain', type=float, help='camera gain (fit only)')
    ap.add_argument('--offset', type=float, help='camera offset (fit only)')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_model(a.weights, device)
    frame = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    f, al = guess(net, [frame], [a.phis], a.radius, a.fsigma)
    print('guess:', ', '.join(f'f = {x:.4f} N, alpha = {np.degrees(y):.1f} deg' for x, y in zip(f[0], al[0])))
    if a.fit:
        if a.gain is None or a.offset is None:
            sys.exit('--fit needs --gain and --offset (the camera intensity scale)')
        f, al = refine([frame], [a.phis], f, al, a.radius, a.fsigma, net['meta']['halo'], a.gain, a.offset, a.setting,
                       device)
        print('fit:  ', ', '.join(f'f = {x:.4f} N, alpha = {np.degrees(y):.1f} deg' for x, y in zip(f[0], al[0])))


if __name__ == '__main__':
    main()
