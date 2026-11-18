"""
Package a trained run into a single weights file for infer.py.

    python export_weights.py <run_dir> <out.pt> --cond clean|halo

The file holds the network state, the target transform and the metadata infer.py needs (architecture, input size,
training fsigma and disk diameter, residual-stress amplitude of the training images).
"""
import argparse
import json
import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate import CFG
from render import HALO_AMP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('out')
    ap.add_argument('--cond', required=True, choices=list(HALO_AMP))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(a.run_dir, 'config.yaml'), encoding='utf8'))
    ck = torch.load(os.path.join(a.run_dir, 'best.pt'), map_location='cpu')
    tf = json.load(open(os.path.join(a.run_dir, 'target_transform.json'), encoding='utf8'))
    meta = dict(cond=a.cond, halo=HALO_AMP[a.cond], halo_power=CFG['halo_power'], hidden=cfg['hidden'],
                dropout=cfg['dropout'], input_size=cfg['input_size'], fsigma_train=CFG['fsigma'], R_ref=CFG['R_ref'],
                px_train=CFG['px_train'], crop_frac=CFG['crop_frac'], val=ck.get('val'))
    torch.save({'model': ck['model'], 'target_transform': tf, 'meta': meta}, a.out)
    print('saved', a.out, meta)


if __name__ == '__main__':
    main()
