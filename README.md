# CRIF

CRIF is a convolutional network that predicts the contact forces on a photoelastic disk, one contact at a time, from
a crop of the disk image around the contact. Its prediction can be used directly, for example for per-particle or
boundary forces, or as the initial guess of a photoelastic force fit: started from it, a fit reaches the accuracy of a
fit started from the true forces, also for strongly loaded disks.

The network is trained only on synthetic images rendered with a forward model. To use it for a different material,
particle size or stress model, render a new training set with your forward model and retrain with the same recipe.

This repository offers the trained networks and everything needed to train your own: the forward model, the synthetic
dataset generator, the training protocol, a minimal image fit and an inference script.

## Contents

| file | purpose |
|---|---|
| `crif/render.py` | forward model: stress field of a disk under point contact forces, optional residual stress, photoelastic intensity |
| `crif/forcesets.py` | random contact-force sets in force and torque balance |
| `crif/generate.py` | synthetic dataset: rendering, camera model (blur, gain, offset, noise), contact crops |
| `crif/data.py`, `model.py` | crop datasets and target transform; ResNet-18 regressor |
| `crif/train.py`, `evaluate.py` | two-phase training (head warm-up, then fine-tuning) and evaluation on held-out disks |
| `crif/fit.py` | minimal image fit started from a guess, with the same forward model as the network |
| `crif/infer.py` | guess (and optional fit) for experimental disk images |
| `crif/export_weights.py` | package a trained run into a weights file for `infer.py` |
| `configs/clean.yaml`, `halo.yaml` | training recipes of the two released networks |
| `tests/test_smoke.py` | quick end-to-end check |

## Install

```
conda env create -f environment.yml
conda activate crif
```

## Trained networks

Two trained networks are attached to the [release](../../releases):

| file | trained on |
|---|---|
| `crif_clean.pt` | disks without residual stress |
| `crif_halo.pt` | disks with a swollen rim (residual-stress amplitude `halo = +1`, see `render.py`) |

Both were trained on disks rendered at fsigma = 225 N/m with a 100 px diameter, forces up to about 6 N on a 6 mm
radius disk, 2–6 contacts per disk and contact angles of attack up to ±40°. Put them in `weights/` and check the
installation:

```
python tests/test_smoke.py
```

Expected on synthetic disks: median force errors of the guess of about 1–1.5% and angle errors below 1°.

## Use on experimental images

```
python crif/infer.py --weights weights/crif_clean.pt --image disk.png \
                        --phis 0.12 1.75 3.30 --radius 0.0075 --fsigma 225
```

- `--image`: grayscale image of one disk, filling the square frame. It is resized to the training diameter (100 px)
  before the crops are cut, so any resolution works; below about 64 px across the guess starts to lose accuracy.
- `--phis`: contact angles in rad, measured in the image as `atan2(row - cy, col - cx)`. Contact detection is not part
  of this repository.
- `--radius`, `--fsigma`: physical disk radius in m and photoelastic constant of the material in N/m. The network
  predicts f/R at the training value of fsigma and the forces are rescaled.
- `--fit --gain G --offset O`: refine the guess with the minimal fit. The fit compares the image on an absolute
  intensity scale, so it needs the camera's gain and offset (`fit.intensity_levels` estimates them from many frames
  of an experiment). It uses the forward model the network was trained for, taken from the weights file.

For many disks, use `infer.guess` and `infer.refine` from Python; both take lists of frames and contact angles.

Choose the network whose training images match your disks. A network applied to disks with a different stress
model (for example the clean network on disks with residual stress) gives poor guesses for weakly loaded contacts.

## About the fit

`fit.py` fits only the image: the loss is the mean squared difference between the measured disk and the rendered
one, with no force- or torque-balance penalty. It is meant to refine and test the guess with a forward model that is
consistent with the network. The guesses can equally be passed to an existing fitting pipeline (for example PeGS),
as long as its forward model matches the network's training images; balance constraints can be added there.

## Train your own network

```
python crif/generate.py                    # about 18,000 disks per condition, a few hours on a GPU
python crif/train.py configs/clean.yaml    # or configs/halo.yaml; about 10 h on a consumer GPU
python crif/evaluate.py runs/<run>
python crif/export_weights.py runs/<run> weights/my_net.pt --cond clean
```

The dataset and the runs go to `data/` and `runs/` next to the repository; set `CRIF_DATASET` and `CRIF_RUNS`
to put them elsewhere. `python crif/generate.py <folder> 200` builds a small dataset for a quick test.

To adapt the network:
- **material and size**: forces are learned as f/R at fsigma = 225 N/m and rescaled at inference, so a different
  fsigma or radius needs no retraining as long as the fringe orders stay within the training range;
- **force range, contact numbers, camera**: change the sampler and camera settings in `generate.CFG`;
- **stress model**: change `render.py` (for example add a residual-stress amplitude to `HALO_AMP`), regenerate and
  retrain. Training sets of order 10⁴–10⁵ contacts gave errors near 1%.

## License

Code: MIT (see `LICENSE`). Trained weights: CC BY 4.0.
