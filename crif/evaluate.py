"""
Evaluate a trained run on the held-out synthetic test sets.

    python evaluate.py <run_dir> [--sources clean/test halo/test clean/test_px80 ...] [--ckpt best.pt]

Default sources: {clean, halo} x {test, test_px64, test_px80, test_px128, test_px160}.
Writes <run_dir>/eval/predictions.parquet (one row per contact and source; the regression output
before any optimizer) and <run_dir>/eval/summary.csv (per source, overall and per force decade).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import CropTensorDataset, TargetTransform, load_contacts, to_network_input
from model import build_model
from paths import DATASET_ROOT

DEFAULT_SOURCES = [f'{c}/{s}' for c in ('clean', 'halo')
                   for s in ('test', 'test_px64', 'test_px80', 'test_px128', 'test_px160')]
PX_OF = {'test': 100, 'test_px64': 64, 'test_px80': 80, 'test_px128': 128, 'test_px160': 160}


@torch.no_grad()
def predict(model, ds, device, cfg, batch=256):
    model.eval()
    out = []
    for s in range(0, len(ds), batch):
        x = torch.from_numpy(ds.images[s:s + batch]).to(device)
        out.append(model(to_network_input(x, cfg['input_size'])).cpu().numpy())
    return np.concatenate(out)


def summarize(df):
    rows = []
    for (src,), g in df.groupby(['source']):
        for label, sel in [('all', np.ones(len(g), bool))] + [
                (f'f in [{lo:g},{hi:g}) N', (g.f >= lo) & (g.f < hi)) for lo, hi in
                [(0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1), (1, 3), (3, 10)]]:
            h = g[sel]
            if len(h) == 0:
                continue
            lr = np.log(h.f_pred / h.f)
            rows.append(dict(source=src, subset=label, n=len(h),
                             med_abs_rel_f=np.median(np.abs(h.f_pred / h.f - 1)),
                             bias_logf=lr.mean(), rms_logf=np.sqrt((lr ** 2).mean()),
                             mae_alpha_deg=np.degrees(np.abs(h.alpha_pred - h.alpha).mean()),
                             r_alpha=np.corrcoef(h.alpha, h.alpha_pred)[0, 1] if len(h) > 2 else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--sources', nargs='*', default=DEFAULT_SOURCES)
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--out-name', default='eval', help='output subfolder of run_dir')
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, 'config.yaml'), encoding='utf8') as fh:
        cfg = yaml.safe_load(fh)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=device)
    tf = TargetTransform(**ck['target_transform'])
    model = build_model(False, cfg['dropout'], cfg['hidden']).to(device)
    model.load_state_dict(ck['model'])

    root = cfg.get('dataset_root', DATASET_ROOT)
    contacts = load_contacts(root)
    R = cfg.get('R_ref', 0.006)
    parts = []
    for src in args.sources:
        ds = CropTensorDataset(root, [src], tf, contacts)
        fR, a = tf.decode(predict(model, ds, device, cfg))
        lab = ds.labels
        parts.append(pd.DataFrame(dict(contact_id=lab.contact_id, disk_id=lab.disk_id, z=lab.z, source=src,
                                       cond=lab.cond, px=PX_OF[src.split('/')[1]], f=lab.f, alpha=lab.alpha,
                                       f_pred=fR * R, alpha_pred=a)))
        print(f'{src}: {len(lab)} contacts', flush=True)
    df = pd.concat(parts, ignore_index=True)
    out = os.path.join(args.run_dir, args.out_name)
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, 'predictions.parquet'), index=False)
    s = summarize(df)
    s.to_csv(os.path.join(out, 'summary.csv'), index=False, float_format='%.4g')
    print(s[s.subset == 'all'].to_string(index=False, float_format=lambda v: f'{v:.4g}'))


if __name__ == '__main__':
    main()
