"""
Serving path: a raw C-MAPSS CSV in, one RUL per engine out.

Deliberately the same arithmetic as the study, reached through the same
functions wherever one exists -- `model.geometry` computes the drift features,
`model.windows` builds the causal window, `model.inference` runs the MC-dropout
passes. A serving path that reimplements any of those would drift from the
reported numbers without anything failing.

The order of operations mirrors the training pipeline exactly:

    raw sensors -> train-fitted z-score (global, or per regime)
                -> geometry against the train-fitted healthy reference
                -> last causal window of W cycles
                -> 5 seeds x MC-dropout passes
                -> the ensembling technique that subset reports

Nothing is fitted here. Every statistic comes from the bundle, so an upload
cannot influence its own normalisation.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from model.config import SCALE, W, MC_PASSES_TEST, RUL_CAP
from model.gaugenet import build_model
from model.geometry import (GEOM_COLS, precompute_riemannian,
                            precompute_wasserstein_mean,
                            precompute_riemannian_personal,
                            precompute_wasserstein_personal)
from model.inference import mc_predict
from model.windows import causal_current_window
from ablation.configs import config_by_name

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(REPO_ROOT, 'app', 'bundles')
CONFIG = config_by_name('sensor_branch_linear_far')

# The 26 columns of a raw C-MAPSS file, in order.
RAW_COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
OP_COLS = ['op1', 'op2', 'op3']
EPS = 1e-8


def available_subsets(bundle_dir=BUNDLE_DIR):
    if not os.path.isdir(bundle_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(bundle_dir) if f.endswith('.pkl'))


def load_bundle(subset, bundle_dir=BUNDLE_DIR):
    path = os.path.join(bundle_dir, f'{subset}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No deployment bundle for {subset} at {path}\n\n"
            f"Build it where the C-MAPSS data lives:\n"
            f"    python -m preprocessing {subset}\n"
            f"    python -m app.bundle {subset}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_cmapss_csv(path_or_buffer):
    """Reads a raw C-MAPSS trajectory file.

    Accepts the whitespace-delimited format NASA ships and ordinary
    comma-separated CSV, with or without a header row, since a user exporting
    the same data from a spreadsheet will produce the latter. Columns are
    matched by name when a header is present and by position when it is not --
    guessing from position on a headed file is how a sensor column silently
    becomes an operating setting.
    """
    for sep in (r'\s+', ','):
        try:
            probe = pd.read_csv(path_or_buffer, sep=sep, header=None, nrows=5,
                                engine='python')
        except Exception:
            continue
        if probe.shape[1] < 26:
            if hasattr(path_or_buffer, 'seek'):
                path_or_buffer.seek(0)
            continue
        if hasattr(path_or_buffer, 'seek'):
            path_or_buffer.seek(0)
        headed = probe.iloc[0].apply(
            lambda v: isinstance(v, str) and not _is_number(v)).any()
        df = pd.read_csv(path_or_buffer, sep=sep, header=0 if headed else None,
                         engine='python')
        if headed:
            df.columns = [str(c).strip().lower() for c in df.columns]
            missing = [c for c in RAW_COLS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Header row is present but these columns are missing: {missing}. "
                    f"Expected the 26 C-MAPSS columns: {RAW_COLS}")
            df = df[RAW_COLS]
        else:
            df = df.iloc[:, :26]
            df.columns = RAW_COLS
        return df.apply(pd.to_numeric, errors='coerce').dropna(subset=['unit', 'cycle'])
    raise ValueError(
        "Could not read 26 C-MAPSS columns from this file. Expected unit, cycle, "
        "op1-op3 and s1-s21, whitespace- or comma-separated.")


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def normalise(df, bundle):
    """Applies the subset's train-fitted normalisation to raw sensor columns.

    Both schemes are read out of the bundle rather than re-derived. For the
    six-regime subsets a row is assigned to the nearest TRAIN-fit centre and
    scaled by that regime's TRAIN statistics -- the same assignment rule the
    preprocessing used for test rows.
    """
    norm = bundle['normalization']
    sensors = norm['sensor_order']
    vals = df[sensors].values.astype(np.float64)

    if norm['scheme'] == 'global':
        out = (vals - norm['trajectory_mu']) / np.maximum(norm['trajectory_sig'], EPS)
        return out, None

    op = (df[OP_COLS].values.astype(np.float64) - norm['op_scaler_mean']) / \
        norm['op_scaler_scale']
    centres = np.asarray(norm['kmeans_centers'])
    regime = np.argmin(((op[:, None, :] - centres[None, :, :]) ** 2).sum(-1), axis=1)

    out = np.zeros_like(vals)
    for r in np.unique(regime):
        mask = regime == r
        if r not in norm['regime_mu']:
            # No train statistics for this centre; leaving the rows at zero
            # would read as a perfectly average sensor, so say so instead.
            raise ValueError(
                f"Rows fell in operating regime {r}, which has no training "
                f"statistics in the {bundle['subset']} bundle.")
        out[mask] = (vals[mask] - norm['regime_mu'][r]) / \
            np.maximum(norm['regime_sig'][r], EPS)
    return out, regime


def attach_geometry(norm_df, bundle):
    """The four drift channels, against the bundle's healthy reference.

    Computed with the study's own functions, so the values an uploaded engine
    gets are produced by the same code that produced the training features.
    Every value at row t uses only cycles <= t of that engine.
    """
    cols = bundle['traj_fn']
    df = norm_df.reset_index(drop=True)
    w = bundle.get('window', W)
    df['riem_global'] = precompute_riemannian(df, cols, bundle['ref_spd'], w)
    df['was_global'] = precompute_wasserstein_mean(df, cols, bundle['was_ref'], w)
    df['riem_personal'] = precompute_riemannian_personal(df, cols, w)
    df['was_personal'] = precompute_wasserstein_personal(df, cols, w)
    return df


def combine(preds_by_seed, bundle):
    """Combines the seeds the way this subset reports.

    Falls back to a plain mean whenever the weights a technique needs are not
    available, and says which technique was actually used -- a weighted number
    silently produced from missing weights would not be the reported one.
    """
    seeds = sorted(preds_by_seed)
    matrix = np.stack([preds_by_seed[s] for s in seeds], axis=0)
    technique = bundle.get('ensemble', 'mean')
    val_nasa = bundle.get('val_nasa')

    if technique == 'mean' or not val_nasa:
        if technique != 'mean':
            technique = f'mean (fell back: no validation scores for {technique})'
        return matrix.mean(axis=0), technique
    if any(s not in val_nasa for s in seeds):
        keep = [i for i, s in enumerate(seeds) if s in val_nasa]
        if not keep:
            return matrix.mean(axis=0), f'mean (fell back: no scored seeds)'
        matrix = matrix[keep]
        seeds = [seeds[i] for i in keep]

    scores = np.array([val_nasa[s] for s in seeds], dtype=np.float64)
    if technique == 'median':
        return np.median(matrix, axis=0), technique
    if technique == 'val_weighted':
        w_ = 1.0 / np.maximum(scores, 1e-6)
    else:
        tau = 50.0 if technique == 'softmax_sharp' else 150.0
        z = -scores / tau
        w_ = np.exp(z - z.max())
    return (matrix * (w_ / w_.sum())[:, None]).sum(axis=0), technique


def predict_csv(path_or_buffer, subset, bundle_dir=BUNDLE_DIR, device=None,
                mc_passes=MC_PASSES_TEST, progress=None):
    """Predicts RUL for every engine in an uploaded C-MAPSS file.

    Returns (DataFrame of per-engine results, info dict).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    bundle = load_bundle(subset, bundle_dir)
    raw = read_cmapss_csv(path_or_buffer)
    if raw.empty:
        raise ValueError("No usable rows found in the uploaded file.")

    norm_vals, regime = normalise(raw, bundle)
    norm_df = raw[['unit', 'cycle']].copy().reset_index(drop=True)
    norm_df[bundle['traj_fn']] = norm_vals
    if progress:
        progress(0.1, "computing geometry features")
    feat_df = attach_geometry(norm_df, bundle)

    fn = list(bundle['traj_fn']) + list(GEOM_COLS)
    fml = bundle['fml']
    w = bundle.get('window', W)

    models = {}
    for seed, rel in bundle['checkpoints'].items():
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint listed in the bundle is missing: {path}")
        m = build_model(len(fn), fml, CONFIG, device=device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models[seed] = m

    units = sorted(feat_df['unit'].unique())
    preds_by_seed = {s: [] for s in models}
    n_cycles = []
    for i, unit in enumerate(units):
        if progress:
            progress(0.15 + 0.8 * i / len(units), f"engine {unit} ({i + 1}/{len(units)})")
        g = feat_df[feat_df['unit'] == unit].sort_values('cycle')
        arr = g[fn].values.astype(np.float32)
        n_cycles.append(len(arr))
        # The last causal window, exactly as the study scores a test engine:
        # everything the engine has shown up to its final observed cycle.
        win = causal_current_window(arr, len(arr) - 1)
        x = torch.from_numpy(win[np.newaxis]).to(device)
        lf = torch.tensor([float(g['cycle'].values[-1] / fml)],
                          dtype=torch.float32).to(device)
        for seed, m in models.items():
            with torch.no_grad():
                preds_by_seed[seed].append(
                    float(mc_predict(m, x, lf, M=mc_passes).item()) * SCALE)

    preds_by_seed = {s: np.array(v) for s, v in preds_by_seed.items()}
    combined, technique = combine(preds_by_seed, bundle)
    spread = np.stack(list(preds_by_seed.values()), axis=0).std(axis=0)

    result = pd.DataFrame({
        'engine': units,
        'cycles_observed': n_cycles,
        'predicted_RUL': np.round(combined, 1),
        'seed_spread_std': np.round(spread, 2),
    })
    for seed in sorted(preds_by_seed):
        result[f'seed_{seed}'] = np.round(preds_by_seed[seed], 1)

    info = dict(subset=subset, device=device, n_engines=len(units),
                n_seeds=len(models), technique=technique, mc_passes=mc_passes,
                window=w, rul_cap=RUL_CAP, scheme=bundle['normalization']['scheme'],
                regimes=(None if regime is None else int(len(np.unique(regime)))))
    return result, info
