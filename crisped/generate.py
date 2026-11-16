"""
Build the synthetic contact-force dataset.

    python generate.py                       full dataset at paths.DATASET_ROOT (about 18,000 disks x 2 conditions)
    python generate.py <out_root> <n>        small dataset with n force sets per contact number, for a quick test

Layout under the output folder:
  config.json                       all parameters + code git hash
  forcesets.parquet                 one row per contact of each unique force set (split assigned here)
  disks.parquet                     one row per rendered disk: set_id, rotation, split, camera-augmentation params
  contacts.parquet                  one row per contact crop: labels f [N], f_over_R [N/m], alpha, phi, R
  renders/{cond}_{split}.npy        raw float16 renders [n_disks, PX, PX] in disk_id order (no augmentation)
  crops/{cond}/{split}/{contact_id:07d}.png
  crops/{cond}/test_px{N}/...       held-out test sets re-rendered at other disk resolutions
cond in {clean, halo}: without residual stress and with the swollen rim (render.HALO_AMP). The same disk (forces,
rotation, camera params, noise seed) is rendered in both.

Splits are assigned per force set before rotation, so rotations and contacts of one set never cross splits.
Crop preprocessing (contact_crops), the same for training and for experimental images: uint8 disk frame ->
3x3 Gaussian (sigma 1) -> divide by max -> circular mask -> rotate contact to top -> top-centre square.
"""
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from forcesets import sample_force_sets
from paths import DATASET_ROOT
from render import HALO_AMP, render_batch

CFG = dict(
    name='crisped_v1',
    out_root=DATASET_ROOT,
    seed=20261021,          # the dataset of the paper
    z_values=[2, 3, 4, 5, 6],
    sets_per_z=3000,
    rotations_per_set=2,
    split_fractions={'train': 0.70, 'val': 0.15, 'test': 0.15},
    # force sampling (forcesets.sample_force_sets), forces in N at R_ref; see forcesets.py docstring
    sampler=dict(m_lo=0.02, m_hi=4.0, sigma_n=0.5, a_max_deg=40.0, mu=1.0, gap_min_deg=53.0,
                 n_min_frac=0.02, f_max=6.4, nc_max=5.0),
    R_ref=0.006,            # m; label f_over_R = f / R_ref is the scale-free target
    fsigma=225.0,
    halo_power=6.0,
    px_train=100,           # disk diameter in px for train/val/test; experimental disks are resized to it
    px_test_extra=[64, 80, 128, 160],
    crop_frac=0.46,         # crop side / disk diameter
    # camera model, sampled per disk, applied before the crop preprocessing
    cam_blur_sigma=[0.5, 1.5],   # px
    cam_gain=[0.6, 1.0],
    cam_offset=[0.0, 0.05],
    cam_noise_sigma=[0.005, 0.03],
    supersample=4,          # render at px*ss, area-average down: camera pixels integrate unresolved fringes
    render_batch=32,
)


def git_state():
    try:
        h = subprocess.check_output(['git', '-C', HERE, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', HERE, 'status', '--porcelain'], text=True).strip())
        return h + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def _sample_chunk(args):
    z, n, seed = args
    s = dict(CFG['sampler'])
    s['a_max'], s['gap_min'] = np.radians(s.pop('a_max_deg')), np.radians(s.pop('gap_min_deg'))
    return sample_force_sets(z, n, np.random.default_rng(seed), R=CFG['R_ref'], fsigma=CFG['fsigma'], **s)


def build_tables(rng_seq):
    """Force sets -> split -> rotated disks -> contacts."""
    set_rows, disk_rows, contact_rows = [], [], []
    set_id = disk_id = contact_id = 0
    for z in CFG['z_values']:
        n, n_chunks = CFG['sets_per_z'], 16
        seeds = rng_seq.spawn(n_chunks)
        sizes = [n // n_chunks + (i < n % n_chunks) for i in range(n_chunks)]
        t = time.time()
        with Pool(min(n_chunks, os.cpu_count())) as pool:
            parts = pool.map(_sample_chunk, [(z, s, sd.generate_state(1)[0]) for s, sd in zip(sizes, seeds)])
        f, phi, alpha = (np.concatenate([p[i] for p in parts]) for i in range(3))
        print(f'z={z}: {n} force sets in {time.time() - t:.0f}s', flush=True)

        rng = np.random.default_rng(rng_seq.spawn(1)[0])
        order = rng.permutation(n)
        cuts = np.cumsum([int(round(CFG['split_fractions'][s] * n)) for s in ('train', 'val')])
        split_of = np.empty(n, dtype=object)
        split_of[order[:cuts[0]]], split_of[order[cuts[0]:cuts[1]]], split_of[order[cuts[1]:]] = 'train', 'val', 'test'

        for i in range(n):
            for k in range(z):
                set_rows.append((set_id, z, split_of[i], k, f[i, k], phi[i, k], alpha[i, k]))
            for r in range(CFG['rotations_per_set']):
                rot = rng.uniform(0, 2 * np.pi)
                cam = dict(blur=rng.uniform(*CFG['cam_blur_sigma']), gain=rng.uniform(*CFG['cam_gain']),
                           offset=rng.uniform(*CFG['cam_offset']), noise=rng.uniform(*CFG['cam_noise_sigma']),
                           noise_seed=int(rng.integers(2**31)))
                disk_rows.append(dict(disk_id=disk_id, set_id=set_id, z=z, split=split_of[i], rot=rot, **cam))
                for k in range(z):
                    contact_rows.append((contact_id, disk_id, set_id, z, split_of[i], k, f[i, k],
                                         f[i, k] / CFG['R_ref'], alpha[i, k], (phi[i, k] + rot) % (2 * np.pi)))
                    contact_id += 1
                disk_id += 1
            set_id += 1

    forcesets = pd.DataFrame(set_rows, columns=['set_id', 'z', 'split', 'k', 'f', 'phi', 'alpha'])
    disks = pd.DataFrame(disk_rows)
    contacts = pd.DataFrame(contact_rows, columns=['contact_id', 'disk_id', 'set_id', 'z', 'split', 'k', 'f',
                                                   'f_over_R', 'alpha', 'phi'])
    return forcesets, disks, contacts


def camera(img, d, px_scale=1.0):
    """Camera model on a float render in [0,1] -> uint8 frame (inside-disk offset, gain, blur, noise)."""
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    c = (img.shape[0] - 1) / 2
    disk = np.hypot(yy - c, xx - c) <= img.shape[0] / 2
    x = d['gain'] * img + d['offset'] * disk
    x = cv2.GaussianBlur(x, (0, 0), d['blur'] * px_scale)
    x = x + np.random.default_rng(d['noise_seed']).normal(0, d['noise'], x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def contact_crops(frame_u8, phis):
    """Network inputs of one disk: frame_u8 [px, px] (disk filling the frame, px = px_train), phis contact angles
    (rad, atan2(row - cy, col - cx)) -> list of uint8 crops with the contact at the top centre."""
    px = frame_u8.shape[0]
    blurred = cv2.GaussianBlur(frame_u8, (3, 3), 1).astype(np.float32)
    m = blurred.max()
    disk = blurred / m if m > 0 else blurred
    yy, xx = np.ogrid[:px, :px]
    disk = disk * ((xx - px // 2) ** 2 + (yy - px // 2) ** 2 <= (px // 2) ** 2)
    side = int(round(CFG['crop_frac'] * px))
    out = []
    for phi in phis:
        M = cv2.getRotationMatrix2D((px // 2, px // 2), np.degrees(phi) + 90, 1.0)
        rot = cv2.warpAffine(disk, M, (px, px), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        x1, x2 = max(px // 2 - side // 2, 0), min(px // 2 + side // 2, px)
        crop = rot[0:min(side, px), x1:x2]
        out.append((crop * 255).clip(0, 255).astype(np.uint8))
    return out


def render_split(contacts, disks, split, cond, px, device):
    """Render all disks of a split at resolution px. Returns float16 [n, px, px] in disk order."""
    d = disks[disks.split == split]
    c = contacts[contacts.split == split]
    zmax = int(d.z.max())
    F, A, B = (np.zeros((len(d), zmax)) for _ in range(3))
    pos = {disk_id: i for i, disk_id in enumerate(d.disk_id)}
    rows = c.disk_id.map(pos).to_numpy()
    F[rows, c.k], A[rows, c.k], B[rows, c.k] = c.f, c.alpha, c.phi
    out = np.zeros((len(d), px, px), dtype=np.float16)
    halo = HALO_AMP[cond]
    bs, ss = CFG['render_batch'], CFG['supersample']
    for s in range(0, len(d), bs):
        hi = render_batch(F[s:s + bs], A[s:s + bs], B[s:s + bs], CFG['R_ref'], px * ss, CFG['fsigma'],
                          halo=halo, power=CFG['halo_power'], device=device)
        out[s:s + bs] = torch.nn.functional.avg_pool2d(hi[:, None], ss)[:, 0].cpu().numpy()
    return d, out


def write_crops(d, renders, contacts, out_dir, px):
    os.makedirs(out_dir, exist_ok=True)
    by_disk = contacts.groupby('disk_id')
    px_scale = px / CFG['px_train']
    for i, row in enumerate(d.itertuples()):
        cs = by_disk.get_group(row.disk_id)
        frame = camera(renders[i].astype(np.float32), row._asdict(), px_scale)
        for cid, crop in zip(cs.contact_id, contact_crops(frame, cs.phi.to_numpy())):
            cv2.imwrite(os.path.join(out_dir, f'{cid:07d}.png'), crop)


def main():
    root = CFG['out_root']
    if os.path.exists(os.path.join(root, 'config.json')):
        sys.exit(f'{root} already has a dataset; refusing to overwrite.')
    os.makedirs(os.path.join(root, 'renders'), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seq = np.random.SeedSequence(CFG['seed'])
    forcesets, disks, contacts = build_tables(seq)
    forcesets.to_parquet(os.path.join(root, 'forcesets.parquet'), index=False)
    disks.to_parquet(os.path.join(root, 'disks.parquet'), index=False)
    contacts.to_parquet(os.path.join(root, 'contacts.parquet'), index=False)
    with open(os.path.join(root, 'config.json'), 'w', encoding='utf8', newline='\n') as fh:
        json.dump({**CFG, 'code_git': git_state(), 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'counts': {s: {'sets': int(forcesets[forcesets.split == s].set_id.nunique()),
                                  'disks': int((disks.split == s).sum()),
                                  'contacts': int((contacts.split == s).sum())}
                              for s in ('train', 'val', 'test')}}, fh, indent=2)
    print(contacts.groupby('split').size().to_string(), flush=True)

    for cond in ('clean', 'halo'):
        render_cond(root, cond, device)
    print('DONE', flush=True)


def render_cond(root, cond, device):
    """Render one image condition into the existing dataset at root (same disks, camera params and noise seeds)."""
    disks = pd.read_parquet(os.path.join(root, 'disks.parquet'))
    contacts = pd.read_parquet(os.path.join(root, 'contacts.parquet'))
    for split in ('train', 'val', 'test'):
        t = time.time()
        d, r = render_split(contacts, disks, split, cond, CFG['px_train'], device)
        np.save(os.path.join(root, 'renders', f'{cond}_{split}.npy'), r)
        write_crops(d, r, contacts, os.path.join(root, 'crops', cond, split), CFG['px_train'])
        print(f'{cond}/{split}: {len(d)} disks in {time.time() - t:.0f}s', flush=True)
    for px in CFG['px_test_extra']:
        d, r = render_split(contacts, disks, 'test', cond, px, device)
        write_crops(d, r, contacts, os.path.join(root, 'crops', cond, f'test_px{px}'), px)
        print(f'{cond}/test_px{px} done', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3:   # small dataset: python generate.py <out_root> <sets_per_z>
        CFG.update(out_root=sys.argv[1], sets_per_z=int(sys.argv[2]), name=CFG['name'] + '_dryrun')
    main()
