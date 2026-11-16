# CRISPeD

CRISPeD is a convolutional network that predicts the contact forces on a photoelastic disk, one contact at a time, from
a crop of the disk image around the contact. Its prediction is a good initial guess for the usual photoelastic force
fit: started from it, the fit reaches the accuracy of a fit started from the true forces, also for strongly loaded
disks where a guess from the gradient-squared method (G²) leads the fit into the wrong fringe order.

The network is trained only on synthetic images rendered with a forward model. Changing the forward model, for
example to include residual stress in the disks, only means rendering a new training set and retraining.

This repository holds the forward model, the synthetic dataset generator, the training protocol, the force fit and an
inference script, as used in the accompanying paper (in preparation).

## Contents

| file | purpose |
|---|---|
| `crisped/render.py` | forward model: stress field of a disk under point contact forces, optional residual stress, photoelastic intensity |
| `crisped/forcesets.py` | random contact-force sets in force and torque balance |
| `crisped/generate.py` | synthetic dataset: rendering, camera model (blur, gain, offset, noise), contact crops |
| `crisped/data.py`, `model.py` | crop datasets and target transform; ResNet-18 regressor |
| `crisped/train.py`, `evaluate.py` | two-phase training (head warm-up, then fine-tuning) and evaluation on held-out disks |
| `crisped/fit.py` | per-disk force fit started from a guess, loose and tight settings |
| `crisped/infer.py` | guess (and optional fit) for experimental disk images |
| `crisped/export_weights.py` | package a trained run into a weights file for `infer.py` |
| `configs/clean.yaml`, `halo.yaml` | training recipes of the two networks in the paper |
| `tests/test_smoke.py` | quick end-to-end check |

## Install

```
conda env create -f environment.yml
conda activate crisped
```

## Pretrained networks

The two networks of the paper are attached to the [release](../../releases) of this version:

| file | trained on | use for |
|---|---|---|
| `crisped_clean.pt` | disks without residual stress | CRISPeD in the paper |
| `crisped_halo.pt` | disks with a swollen rim (residual-stress amplitude `halo = +1`, see `render.py`) | halo-CRISPeD in the paper |

Put them in `weights/` and check the installation:

```
python tests/test_smoke.py
```

Expected: median force errors of about 1–1.5% and angle errors below 1° for both networks.

## Use on experimental images

```
python crisped/infer.py --weights weights/crisped_halo.pt --image disk.png \
                        --phis 0.12 1.75 3.30 --radius 0.0075 --fsigma 225
```

- `--image`: grayscale image of one disk, filling the square frame. It is resized to the training diameter (100 px)
  before the crops are cut, so any resolution works; below about 64 px across the guess starts to lose accuracy.
- `--phis`: contact angles in rad, measured in the image as `atan2(row - cy, col - cx)`.
- `--radius`, `--fsigma`: physical disk radius in m and photoelastic constant of the material in N/m. The network
  predicts f/R at the training value of fsigma (225 N/m) and the forces are rescaled.
- `--fit --gain G --offset O`: refine the guess with the force fit. The fit compares the image on an absolute
  intensity scale, so it needs the camera's gain and offset (`fit.intensity_levels` estimates them from many frames
  of an experiment). Use the forward model the network was trained for: `infer.py` takes it from the weights file.

For many disks, use `infer.guess` and `infer.refine` from Python; both take lists of frames and contact angles.

## Retrain

```
python crisped/generate.py                    # about 18,000 disks per condition, a few hours on a GPU
python crisped/train.py configs/clean.yaml    # or configs/halo.yaml; about 10 h on a consumer GPU
python crisped/evaluate.py runs/<run>
python crisped/export_weights.py runs/<run> weights/my_net.pt --cond clean
```

The dataset and the runs go to `data/` and `runs/` next to the repository; set `CRISPED_DATASET` and `CRISPED_RUNS`
to put them elsewhere. `python crisped/generate.py <folder> 200` builds a small dataset for a quick test. With the
default seed, `generate.py` reproduces the dataset of the paper.

To train for a different forward model, change `render.py` (or add an amplitude to `HALO_AMP`), regenerate and
retrain with the same recipe. In the paper, training sets of order 10⁴–10⁵ contacts were needed for errors near 1%.

## License

Code: MIT (see `LICENSE`). Pretrained weights: CC BY 4.0.
