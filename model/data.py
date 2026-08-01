"""
Loads a preprocessed subset and attaches the geometry features.

Shared by training and evaluation so both stages always see an identical
feature matrix in an identical column order -- a checkpoint is only valid
against the feature list it was trained on.
"""

import os
import pickle

import numpy as np
import pandas as pd

from .config import DATA_ROOT, OUT_ROOT, RUL_CAP, PIPELINE_VERSION
from .geometry import build_geometry


def pipeline_path(subset):
    version = PIPELINE_VERSION.get(subset, 'v1')
    return os.path.join(DATA_ROOT, 'pipeline_output', f'output_{subset}_{version}.pkl')


def load_subset(subset, cache_dir=None):
    """Returns (fn, tr_c, te_c, rul_true, fml).

    fn        ordered feature columns: trajectory + causal baseline + geometry
    tr_c/te_c train and test frames with the geometry columns attached
    rul_true  ground-truth test RUL, capped to match the training target
    fml       median fleet life, the scale that normalises cycle -> life
              fraction; taken from TRAIN only, since test engines are
              truncated and their observed length is not their life
    """
    cache_dir = cache_dir or os.path.join(OUT_ROOT, subset)
    os.makedirs(cache_dir, exist_ok=True)

    with open(pipeline_path(subset), 'rb') as f:
        out = pickle.load(f)
    traj_fn = out['trajectory_feature_names'][subset]
    base_fn = out['baseline_feature_names'][subset]
    tr_raw = out['feature_data'][subset]['train']
    te_raw = out['feature_data'][subset]['test']

    geom_cols, tr_c, te_c = build_geometry(traj_fn, tr_raw, te_raw, cache_dir)
    fn = traj_fn + base_fn + geom_cols

    rul_file = os.path.join(DATA_ROOT, f'RUL_{subset}.txt')
    rul_true = np.minimum(
        pd.read_csv(rul_file, header=None, names=['rul']).values.ravel(), RUL_CAP)
    fml = float(np.median(tr_c.groupby('unit')['cycle'].max().values))
    return fn, tr_c, te_c, rul_true, fml
