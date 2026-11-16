"""
Train the contact-force regressor (CRISPeD) on the synthetic contact crops.

    python train.py configs/clean.yaml [--run-id my_run] [--resume]

The dataset comes from generate.py (paths.DATASET_ROOT unless the config sets dataset_root).
Run directory <runs_root>/<run_id>_<config name>/ holds:
  config.yaml          resolved config + code git hash + dataset config
  target_transform.json
  log.csv              one row per epoch (phase, lr, train/val losses and errors, time)
  best.pt / last.pt    best-val and latest checkpoints (last.pt also has optimizer state for --resume)
Two phases: warmup (frozen backbone, head only) then fine-tune (all layers),
each with its own LR, epoch budget and early stopping on val loss.
Per phase, `schedule` is 'cosine' (default; LR annealed to lr/100 over `epochs`) or 'plateau'
(LR x `factor` after `sched_patience` epochs without improvement; the phase ends when the LR would
drop below `min_lr`, on early stopping, or at `epochs`). Improvement means val loss below
best * (1 - min_rel_delta) when `min_rel_delta` is set, else below best - min_delta.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import CropTensorDataset, TargetTransform, load_contacts, to_network_input
from model import build_model, set_backbone_trainable
from paths import DATASET_ROOT, RUNS_ROOT


def git_state():
    try:
        h = subprocess.check_output(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', HERE, 'status', '--porcelain', '--', '*.py', 'configs'],
                                        text=True).strip()
        return h + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def batch_errors(y_pred, y_true, tf):
    """Physical errors on a batch: |log(f_pred/f_true)| and |alpha_pred - alpha_true| (rad)."""
    fp, ap = tf.decode(y_pred)
    ft, at = tf.decode(y_true)
    return np.abs(np.log(np.maximum(fp, 1e-9) / ft)), np.abs(ap - at)


def run_epoch(model, loader, device, cfg, tf, optimizer=None):
    train = optimizer is not None
    model.train(train)
    w = torch.tensor([cfg['loss']['w_force'], cfg['loss']['w_alpha']], device=device)
    tot, n, logf_err, a_err = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for imgs, y, _ in loader:
            imgs, y = imgs.to(device, non_blocking=True), y.to(device, non_blocking=True)
            flip = None
            if train and cfg['augment']['hflip']:
                flip = torch.rand(len(y), device=device) < 0.5
                y = y.clone()
                y[flip, 1] = -y[flip, 1]        # mirror about the contact's radial axis: alpha -> -alpha
            out = model(to_network_input(imgs, cfg['input_size'], flip))
            per = torch.nn.functional.smooth_l1_loss(out, y, reduction='none', beta=cfg['loss']['huber_beta']) \
                if cfg['loss']['kind'] == 'huber' else (out - y) ** 2
            loss = (per * w).sum(1).mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            tot += loss.item() * len(y)
            n += len(y)
            e1, e2 = batch_errors(out.detach().cpu().numpy(), y.cpu().numpy(), tf)
            logf_err.append(e1)
            a_err.append(e2)
    logf_err, a_err = np.concatenate(logf_err), np.concatenate(a_err)
    return dict(loss=tot / n, med_rel_f=float(np.median(np.expm1(logf_err))),
                rms_logf=float(np.sqrt(np.mean(logf_err ** 2))), mae_alpha_deg=float(np.degrees(a_err.mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--run-id', default=time.strftime('%Y%m%d'), help='prefix of the run directory (default: date)')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    with open(args.config, encoding='utf8') as fh:
        cfg = yaml.safe_load(fh)
    name = os.path.splitext(os.path.basename(args.config))[0]
    cfg.setdefault('dataset_root', DATASET_ROOT)
    cfg.setdefault('runs_root', RUNS_ROOT)
    run_dir = os.path.join(cfg['runs_root'], f'{args.run_id}_{name}')
    if os.path.exists(os.path.join(run_dir, 'log.csv')) and not args.resume:
        sys.exit(f'{run_dir} exists; use --resume or a new --run-id')
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    root = cfg['dataset_root']
    contacts = load_contacts(root)
    train_lab = contacts[contacts.split == 'train']
    tf = TargetTransform(cfg['force_target']).fit(train_lab.f_over_R.to_numpy())
    with open(os.path.join(root, 'config.json'), encoding='utf8') as fh:
        ds_cfg = json.load(fh)

    if not args.resume:
        with open(os.path.join(run_dir, 'config.yaml'), 'w', encoding='utf8', newline='\n') as fh:
            yaml.safe_dump({**cfg, 'run_id': args.run_id, 'config_file': name, 'code_git': git_state(),
                            'dataset_name': ds_cfg['name'], 'dataset_code_git': ds_cfg.get('code_git'),
                            'started': time.strftime('%Y-%m-%d %H:%M:%S')}, fh, sort_keys=False)
        with open(os.path.join(run_dir, 'target_transform.json'), 'w', encoding='utf8', newline='\n') as fh:
            json.dump(tf.state(), fh, indent=2)

    mx = cfg.get('max_items')
    ds_train = CropTensorDataset(root, cfg['train_sources'], tf, contacts, mx, cfg['seed'])
    ds_val = CropTensorDataset(root, cfg['val_sources'], tf, contacts, mx and max(mx // 4, 256), cfg['seed'])
    mk = lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle,
                                                         num_workers=0, pin_memory=True, drop_last=shuffle)
    dl_train, dl_val = mk(ds_train, True), mk(ds_val, False)
    print(f'train {len(ds_train)}  val {len(ds_val)}  -> {run_dir}', flush=True)

    model = build_model(cfg['pretrained'], cfg['dropout'], cfg['hidden']).to(device)
    log_path = os.path.join(run_dir, 'log.csv')
    state = dict(phase_idx=0, epoch_in_phase=0, best_val=math.inf, bad_epochs=0, global_epoch=0)
    ckpt_last = os.path.join(run_dir, 'last.pt')
    resume_opt = None
    if args.resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck['model'])
        state, resume_opt = ck['state'], ck['optimizer']
        print(f'resumed at phase {state["phase_idx"]} epoch {state["epoch_in_phase"]}', flush=True)

    fields = ['global_epoch', 'phase', 'epoch', 'lr', 'train_loss', 'val_loss', 'val_med_rel_f', 'val_rms_logf',
              'val_mae_alpha_deg', 'train_med_rel_f', 'seconds']
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as fh:
            csv.writer(fh).writerow(fields)

    phases = cfg['phases']
    while state['phase_idx'] < len(phases):
        ph = phases[state['phase_idx']]
        set_backbone_trainable(model, ph['train_backbone'])
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=ph['lr'], weight_decay=ph.get('weight_decay', 0.0))
        plateau = ph.get('schedule', 'cosine') == 'plateau'
        if plateau:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=ph['factor'], patience=ph['sched_patience'], threshold=ph.get('min_rel_delta', 1e-4),
                threshold_mode='rel', min_lr=0.0)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ph['epochs'], eta_min=ph['lr'] * 0.01)
        if resume_opt is not None:
            opt.load_state_dict(resume_opt['opt'])
            sched.load_state_dict(resume_opt['sched'])
            resume_opt = None
        while state['epoch_in_phase'] < ph['epochs']:
            t0 = time.time()
            tr = run_epoch(model, dl_train, device, cfg, tf, opt)
            va = run_epoch(model, dl_val, device, cfg, tf)
            lr = opt.param_groups[0]['lr']
            sched.step(va['loss']) if plateau else sched.step()
            state['epoch_in_phase'] += 1
            state['global_epoch'] += 1
            row = [state['global_epoch'], ph['name'], state['epoch_in_phase'], f'{lr:.3g}', f'{tr["loss"]:.5f}',
                   f'{va["loss"]:.5f}', f'{va["med_rel_f"]:.4f}', f'{va["rms_logf"]:.4f}',
                   f'{va["mae_alpha_deg"]:.3f}', f'{tr["med_rel_f"]:.4f}', f'{time.time() - t0:.0f}']
            with open(log_path, 'a', newline='') as fh:
                csv.writer(fh).writerow(row)
            print(' '.join(f'{k}={v}' for k, v in zip(fields, row)), flush=True)

            if 'min_rel_delta' in ph:
                improved = va['loss'] < state['best_val'] * (1 - ph['min_rel_delta'])
            else:
                improved = va['loss'] < state['best_val'] - ph['min_delta']
            if improved:
                state['best_val'], state['bad_epochs'] = va['loss'], 0
                torch.save({'model': model.state_dict(), 'state': dict(state), 'val': va,
                            'target_transform': tf.state()}, os.path.join(run_dir, 'best.pt'))
            else:
                state['bad_epochs'] += 1
            torch.save({'model': model.state_dict(), 'state': dict(state),
                        'optimizer': {'opt': opt.state_dict(), 'sched': sched.state_dict()}}, ckpt_last)
            if state['bad_epochs'] >= ph['patience']:
                print(f'early stop in phase {ph["name"]}', flush=True)
                break
            if plateau and opt.param_groups[0]['lr'] < ph['min_lr']:
                print(f'LR below min_lr in phase {ph["name"]}: converged', flush=True)
                break
        # next phase starts from the best weights so far
        model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), map_location=device)['model'])
        state.update(phase_idx=state['phase_idx'] + 1, epoch_in_phase=0, bad_epochs=0)
        torch.save({'model': model.state_dict(), 'state': dict(state), 'optimizer': None}, ckpt_last)
    print('TRAINING DONE', flush=True)


if __name__ == '__main__':
    main()
