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

DOWNLOAD_HINT = (
    "Download the C-MAPSS archive from the NASA Prognostics Data Repository\n"
    "  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/\n"
    "and extract train_<subset>.txt, test_<subset>.txt and RUL_<subset>.txt into\n"
    "the data directory (or point --data-root / CMAPPS_DATA_ROOT elsewhere)."
)


def pipeline_path(subset, data_root=None):
    version = PIPELINE_VERSION.get(subset, 'v1')
    return os.path.join(data_root or DATA_ROOT, 'pipeline_output',
                        f'output_{subset}_{version}.pkl')


def raw_path(subset, data_root=None):
    return os.path.join(data_root or DATA_ROOT, f'RUL_{subset}.txt')


REGIME_COL = 'regime_id'


def features_for(cfg, traj_fn, geom_cols):
    """The ordered feature list one config trains on.

    Geometry is a rung of the ladder, not a fixed part of the input, so the
    columns are attached to the frames once and selected per config here --
    the alternative, loading the subset twice, would recompute the geometry
    cache for no reason.

    regime_id, when the config asks for it, goes LAST and stays last: the
    model splits it off the feature tensor by position, so putting it anywhere
    else would feed a sensor into the embedding and the regime id into the
    projection, with no error to say so.
    """
    fn = list(traj_fn) + (list(geom_cols) if cfg['use_geometry'] else [])
    if cfg.get('use_regime'):
        fn.append(REGIME_COL)
    return fn


def load_subset(subset, cache_dir=None, data_root=None, out_root=None):
    """Returns (traj_fn, geom_cols, tr_c, te_c, rul_true, fml).

    traj_fn   the trajectory sensor columns, in order
    geom_cols the geometry columns, attached to the frames but NOT prepended
              to a feature list here: pass both through features_for(cfg, ...)
    tr_c/te_c train and test frames with the geometry columns attached
    rul_true  ground-truth test RUL, capped to match the training target
    fml       median fleet life, the scale that normalises cycle -> life
              fraction; taken from TRAIN only, since test engines are
              truncated and their observed length is not their life
    """
    data_root = data_root or DATA_ROOT
    cache_dir = cache_dir or os.path.join(out_root or OUT_ROOT, subset)
    os.makedirs(cache_dir, exist_ok=True)

    pkl = pipeline_path(subset, data_root)
    if not os.path.exists(pkl):
        raise FileNotFoundError(
            f"No preprocessed input at {pkl}\n\n"
            f"Run the preprocessing stage first:\n"
            f"    python -m preprocessing {subset} --data-root {data_root}\n\n"
            f"{DOWNLOAD_HINT}")

    rul_file = raw_path(subset, data_root)
    if not os.path.exists(rul_file):
        raise FileNotFoundError(
            f"Missing ground-truth file {rul_file}\n\n{DOWNLOAD_HINT}")

    with open(pkl, 'rb') as f:
        out = pickle.load(f)
    traj_fn = out['trajectory_feature_names'][subset]
    # Baseline features were ablated out (worse TEST_NASA on both subsets), so
    # the final pipeline emits an empty list here. An older pickle would still
    # carry 14 baseline columns, which would silently change the feature count
    # a checkpoint was trained against -- so say what was found rather than
    # concatenating whatever turns up.
    base_fn = out['baseline_feature_names'][subset]
    if base_fn:
        print(f"  WARNING: {os.path.basename(pkl)} carries {len(base_fn)} baseline "
              f"features; the final pipeline emits none. They are being INCLUDED -- "
              f"regenerate the pickle to drop them.")
    tr_raw = out['feature_data'][subset]['train']
    te_raw = out['feature_data'][subset]['test']

    has_regime = REGIME_COL in tr_raw.columns and REGIME_COL in te_raw.columns
    geom_cols, tr_c, te_c = build_geometry(traj_fn, tr_raw, te_raw, cache_dir)
    traj_fn = traj_fn + base_fn
    print(f"  features: {len(traj_fn)} sensors + {len(geom_cols)} geometry available "
          f"= {len(traj_fn) + len(geom_cols)} (a config selects geometry or not)"
          f"{'  + regime_id (multi-regime subset)' if has_regime else ''}")

    rul_true = np.minimum(
        pd.read_csv(rul_file, header=None, names=['rul']).values.ravel(), RUL_CAP)
    fml = float(np.median(tr_c.groupby('unit')['cycle'].max().values))
    return traj_fn, geom_cols, tr_c, te_c, rul_true, fml


def has_regimes(tr_c):
    """Whether this subset carries a regime id, i.e. was regime-normalised.

    Read off the frame rather than a subset name, so a rung that conditions on
    the regime is skipped wherever the column genuinely is not there.
    """
    return REGIME_COL in tr_c.columns
