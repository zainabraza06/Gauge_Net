"""
Geometry features: how far a sliding window's sensor covariance / marginal
distribution has drifted from a healthy reference.

Four features are produced per row, in two flavours:

  *_global    reference = the healthy portion pooled over all TRAIN engines.
              Fleet-relative: says how degraded this engine looks compared to
              a typical healthy engine.
  *_personal  reference = this engine's OWN history up to the current cycle.
              Unit-relative: says how far this engine has drifted from how it
              itself started, which cancels unit-to-unit build variation.

  riem_*      affine-invariant Riemannian distance between correlation
              matrices, ||log(R^-1/2 C R^-1/2)||_F -- sensitive to changes in
              the coupling BETWEEN sensors, not their individual levels.
  was_*       1-D Wasserstein-1 distance per sensor, averaged -- sensitive to
              changes in each sensor's own marginal distribution.

Every value at row t uses only cycles <= t of that engine, so the features are
usable in a streaming deployment and carry no future information. The personal
variants are additionally scaled by a reliability factor (how much history is
available yet), so early rows, whose reference is estimated from a handful of
cycles, contribute a damped signal rather than noise.
"""

import os
import pickle

import numpy as np
from scipy.linalg import logm as _scipy_logm, sqrtm as _scipy_sqrtm

from .config import W, HEALTHY_FRAC

GEOM_COLS = ['riem_global', 'was_global', 'riem_personal', 'was_personal']


def _safe_corr_cov(X, eps=1e-6):
    """Correlation matrix with a ridge on the diagonal. Standardising first
    means a sensor's scale drops out and only its co-movement counts;
    the eps ridge keeps the matrix positive definite for logm/sqrtm when a
    window is short or a sensor is constant across it."""
    Xc = X - X.mean(0)
    std = Xc.std(0)
    std = np.where(std < eps, eps, std)
    Xs = Xc / std
    return (Xs.T @ Xs) / max(len(X) - 1, 1) + eps * np.eye(X.shape[1])


def _causal_healthy_slice(g_sorted, cols, frac=HEALTHY_FRAC):
    """Leading frac of one engine's trajectory, treated as healthy."""
    nh = max(2, int(len(g_sorted) * frac))
    return g_sorted.iloc[:nh][cols].values.astype(np.float64)


def compute_ref_spd_single(tr_df, cols):
    """Fleet healthy reference correlation matrix, pooled over TRAIN only."""
    all_h = [_causal_healthy_slice(g.sort_values('cycle'), cols)
             for _, g in tr_df.groupby('unit', sort=False)]
    return _safe_corr_cov(np.vstack(all_h))


def compute_was_ref_single(tr_df, cols):
    """Fleet healthy reference marginal per sensor, pooled over TRAIN only."""
    ref = {s: [] for s in cols}
    for _, g in tr_df.groupby('unit', sort=False):
        h = _causal_healthy_slice(g.sort_values('cycle'), cols)
        for i, s in enumerate(cols):
            ref[s].extend(h[:, i].tolist())
    return {s: np.sort(np.array(v)) for s, v in ref.items()}


def precompute_riemannian(df, cols, ref_spd, w=W):
    """Riemannian distance from the fleet healthy reference, per row."""
    ref_sqrt = _scipy_sqrtm(ref_spd).real
    try:
        ri = np.linalg.inv(ref_sqrt)
    except Exception:
        ri = np.eye(ref_spd.shape[0])
    vals = np.zeros(len(df))
    for _, g in df.groupby('unit', sort=False):
        g_s = g.sort_values('cycle'); idx = g_s.index.values
        X = g_s[cols].values.astype(np.float64)
        for t in range(len(X)):
            win = X[max(0, t - w + 1):t + 1]
            if len(win) < 3:
                vals[idx[t]] = 0.
                continue
            M = ri @ _safe_corr_cov(win) @ ri
            try:
                vals[idx[t]] = float(np.linalg.norm(_scipy_logm(M).real, 'fro'))
            except Exception:
                vals[idx[t]] = 0.
    return vals


def precompute_wasserstein_mean(df, cols, was_ref, w=W):
    """Mean per-sensor Wasserstein-1 distance from the fleet healthy
    reference, per row. Comparing sorted samples against equally spaced
    quantiles of the reference is the 1-D Wasserstein-1 distance."""
    vals = np.zeros(len(df))
    for _, g in df.groupby('unit', sort=False):
        g_s = g.sort_values('cycle'); idx = g_s.index.values
        X = g_s[cols].values.astype(np.float64)
        for t in range(len(X)):
            win = X[max(0, t - w + 1):t + 1]; nw = len(win)
            per_sensor = np.zeros(len(cols))
            for si, sc in enumerate(cols):
                x = np.sort(win[:, si])
                q = np.quantile(was_ref[sc], np.linspace(0, 1, nw))
                per_sensor[si] = float(np.mean(np.abs(x - q)))
            vals[idx[t]] = per_sensor.mean()
    return vals


def precompute_riemannian_personal(df, cols, w=W):
    """Riemannian distance from THIS engine's own early history."""
    vals = np.zeros(len(df))
    for _, g in df.groupby('unit', sort=False):
        g_s = g.sort_values('cycle'); idx = g_s.index.values
        X = g_s[cols].values.astype(np.float64); n = len(X)
        for t in range(n):
            n_ref = min(t + 1, w)
            reliability = min(n_ref / w, 1.0)
            ref_window = X[:n_ref]
            current_window = X[max(0, t - w + 1):t + 1]
            if n_ref < 3 or len(current_window) < 3:
                vals[idx[t]] = 0.
                continue
            try:
                ri = np.linalg.inv(_scipy_sqrtm(_safe_corr_cov(ref_window)).real)
                M = ri @ _safe_corr_cov(current_window) @ ri
                vals[idx[t]] = float(np.linalg.norm(_scipy_logm(M).real, 'fro')) * reliability
            except Exception:
                vals[idx[t]] = 0.
    return vals


def precompute_wasserstein_personal(df, cols, w=W):
    """Mean per-sensor Wasserstein-1 distance from THIS engine's own early
    history."""
    vals = np.zeros(len(df))
    for _, g in df.groupby('unit', sort=False):
        g_s = g.sort_values('cycle'); idx = g_s.index.values
        X = g_s[cols].values.astype(np.float64); n = len(X)
        for t in range(n):
            n_ref = min(t + 1, w)
            reliability = min(n_ref / w, 1.0)
            ref_window = X[:n_ref]
            current_window = X[max(0, t - w + 1):t + 1]
            nw_cur = len(current_window)
            if n_ref < 3 or nw_cur < 3:
                vals[idx[t]] = 0.
                continue
            per_sensor = np.zeros(len(cols))
            for si in range(len(cols)):
                ref_sorted = np.sort(ref_window[:, si])
                cur_sorted = np.sort(current_window[:, si])
                q = np.quantile(ref_sorted, np.linspace(0, 1, nw_cur))
                per_sensor[si] = float(np.mean(np.abs(cur_sorted - q)))
            vals[idx[t]] = per_sensor.mean() * reliability
    return vals


def build_geometry(traj_fn, tr_c, te_c, cache_dir):
    """Appends the four geometry columns to train and test.

    Results are cached: the computation is O(rows x W x sensors^3) and takes
    far longer than a training run, and every seed and config of a dataset
    consumes exactly the same features. Both fleet references are fitted on
    TRAIN only, so no test information reaches the features.
    """
    cache_path = os.path.join(cache_dir, 'geom_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache['geom_cols'], cache['tr_ext'], cache['te_ext']

    ref_spd = compute_ref_spd_single(tr_c, traj_fn)
    was_ref = compute_was_ref_single(tr_c, traj_fn)
    tr_ext = tr_c.copy(); te_ext = te_c.copy()
    tr_ext['riem_global'] = precompute_riemannian(tr_c, traj_fn, ref_spd)
    te_ext['riem_global'] = precompute_riemannian(te_c, traj_fn, ref_spd)
    tr_ext['was_global'] = precompute_wasserstein_mean(tr_c, traj_fn, was_ref)
    te_ext['was_global'] = precompute_wasserstein_mean(te_c, traj_fn, was_ref)
    tr_ext['riem_personal'] = precompute_riemannian_personal(tr_c, traj_fn)
    te_ext['riem_personal'] = precompute_riemannian_personal(te_c, traj_fn)
    tr_ext['was_personal'] = precompute_wasserstein_personal(tr_c, traj_fn)
    te_ext['was_personal'] = precompute_wasserstein_personal(te_c, traj_fn)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(dict(geom_cols=GEOM_COLS, tr_ext=tr_ext, te_ext=te_ext), f)
    return GEOM_COLS, tr_ext, te_ext
