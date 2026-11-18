"""Locations of the synthetic dataset and of the training runs. Defaults are folders next to the repository;
override with the environment variables CRIF_DATASET and CRIF_RUNS, or with dataset_root / runs_root in a
training config."""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get('CRIF_DATASET', os.path.join(_REPO, 'data', 'crif_v1'))
RUNS_ROOT = os.environ.get('CRIF_RUNS', os.path.join(_REPO, 'runs'))
