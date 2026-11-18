"""
In-memory contact-crop datasets.

Crops are small (46x46 px at the 100 px training resolution), so a split is loaded once into a
uint8 array and cached as .npy next to the dataset; resizing to the network input and ImageNet
normalisation happen on the GPU (to_network_input), keeping the CPU free.

Sources are named "<cond>/<split>", e.g. "halo/train", "clean/test_px80".
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_contacts(root):
    return pd.read_parquet(os.path.join(root, 'contacts.parquet'))


def _base_split(split):
    return 'test' if split.startswith('test') else split


def load_source(root, source, contacts=None):
    """Returns (images uint8 [N, h, w], labels DataFrame aligned with images)."""
    cond, split = source.split('/')
    contacts = load_contacts(root) if contacts is None else contacts
    lab = contacts[contacts.split == _base_split(split)].sort_values('contact_id').reset_index(drop=True)
    cache = os.path.join(root, 'cache', f'{cond}_{split}.npy')
    if os.path.exists(cache):
        imgs = np.load(cache)
    else:
        d = os.path.join(root, 'crops', cond, split)
        first = cv2.imread(os.path.join(d, f'{lab.contact_id[0]:07d}.png'), cv2.IMREAD_GRAYSCALE)
        imgs = np.empty((len(lab), *first.shape), dtype=np.uint8)
        for i, cid in enumerate(lab.contact_id):
            imgs[i] = cv2.imread(os.path.join(d, f'{cid:07d}.png'), cv2.IMREAD_GRAYSCALE)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, imgs)
    lab = lab.assign(cond=cond, source=source)
    return imgs, lab


class TargetTransform:
    """Network targets: y0 = standardised log(f/R) (or f/R if force_target='linear'), y1 = alpha / alpha_scale.
    Statistics are fitted on the training labels and stored with the run."""

    def __init__(self, force_target='log', mean=None, std=None, alpha_scale=np.pi / 4):
        self.force_target, self.mean, self.std, self.alpha_scale = force_target, mean, std, alpha_scale

    def _raw(self, f_over_R):
        return np.log(f_over_R) if self.force_target == 'log' else np.asarray(f_over_R)

    def fit(self, f_over_R):
        r = self._raw(f_over_R)
        self.mean, self.std = float(r.mean()), float(r.std())
        return self

    def encode(self, f_over_R, alpha):
        return np.stack([(self._raw(f_over_R) - self.mean) / self.std, np.asarray(alpha) / self.alpha_scale], 1)

    def decode(self, y):
        """y [N, 2] -> f_over_R, alpha."""
        r = y[:, 0] * self.std + self.mean
        return (np.exp(r) if self.force_target == 'log' else r), y[:, 1] * self.alpha_scale

    def state(self):
        return dict(force_target=self.force_target, mean=self.mean, std=self.std, alpha_scale=self.alpha_scale)


class CropTensorDataset(torch.utils.data.Dataset):
    """Concatenated sources; returns (uint8 crop [h, w], target [2], row index)."""

    def __init__(self, root, sources, target_tf, contacts=None, max_items=None, seed=0):
        contacts = load_contacts(root) if contacts is None else contacts
        parts = [load_source(root, s, contacts) for s in sources]
        shapes = {p[0].shape[1:] for p in parts}
        if len(shapes) != 1:
            raise ValueError(f'sources have different crop sizes {shapes}; evaluate them separately')
        self.images = np.concatenate([p[0] for p in parts])
        self.labels = pd.concat([p[1] for p in parts], ignore_index=True)
        if max_items is not None and max_items < len(self.labels):
            keep = np.sort(np.random.default_rng(seed).choice(len(self.labels), max_items, replace=False))
            self.images, self.labels = self.images[keep], self.labels.iloc[keep].reset_index(drop=True)
        self.targets = target_tf.encode(self.labels.f_over_R.to_numpy(), self.labels.alpha.to_numpy()).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(self.images[i]), torch.from_numpy(self.targets[i]), i


def to_network_input(batch_u8, size=224, hflip_mask=None):
    """uint8 [B, h, w] on device -> normalised float [B, 3, size, size].
    Equivalent to torchvision Resize -> Grayscale(3) -> ToTensor -> Normalize(ImageNet).
    hflip_mask [B] bool mirrors crops left-right (the caller must negate alpha for those rows)."""
    x = batch_u8.float().div_(255.0).unsqueeze(1)
    if hflip_mask is not None and hflip_mask.any():
        x[hflip_mask] = x[hflip_mask].flip(-1)
    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x.expand(-1, 3, -1, -1) - mean) / std
