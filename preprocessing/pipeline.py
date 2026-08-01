"""
C-MAPSS preprocessing -- causal and leak-free, one implementation for FD001
and FD003.

    python -m preprocessing FD001
    python -m preprocessing FD003 --data-root /path/to/cmapps

Both subsets have a SINGLE operating condition, so neither needs the regime
clustering FD002/FD004 require, and both use the same 14 literature-confirmed
sensors. FD003 has two fault modes against FD001's one, but no fault-mode
label is used anywhere: the model is expected to infer the mode from the
trajectory. The methods are therefore identical; see EMIT_PHASE for the one
behavioural difference that is carried over deliberately.

Three properties this pipeline guarantees:

  no train/test leakage   every normalisation statistic (sensor mean/std,
                          baseline mean/std) is fitted on TRAIN only and
                          applied unchanged to test.

  no within-engine leakage  the per-engine baseline at cycle t is an
                          EXPANDING mean over that engine's cycles
                          1..min(t, BASELINE_WINDOW). An earlier revision
                          computed one baseline from each engine's first 30
                          cycles and attached it to EVERY row of that engine,
                          including rows at cycle 1, 5 and 15 -- rows that,
                          at that point in the trajectory, were seeing cycles
                          that had not happened yet. The expanding form is
                          what a streaming deployment could actually compute.

  visible estimate quality  baseline_n_obs records how many cycles each row's
                          baseline is built from, so early rows being a weak
                          estimate is explicit in the data rather than hidden.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
EPS = 1e-8

# The 14 sensors that vary with degradation under a single operating
# condition; the other 7 are constant or pure noise on these subsets.
SELECTED = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
            's13', 's14', 's15', 's17', 's20', 's21']

RUL_CAP = 125.0
BASELINE_WINDOW = 30

# sample_weight shaping (a convenience column for downstream experiments;
# the ablation trainer computes its own phase weight and ignores this).
W_MAX = 3.0
GAMMA = 2.0

VERSION = {'FD001': 'v1', 'FD003': 'v4'}

# Whether to emit onset / T_i / phase.
#
# This is the ONE place the two subsets genuinely differ, and it is a
# historical difference rather than a methodological one: the FD001 pipeline
# emitted these columns, the FD003 v4 pipeline did not. It matters, because
# the ablation trainer's phase weight falls back to onset=0, T_i=engine
# length, phase=1 when they are missing -- which weights the whole trajectory
# on a ramp rather than only the post-onset degradation phase. So FD001 and
# FD003 checkpoints were trained under DIFFERENT loss weightings.
#
# The defaults reproduce what each pipeline actually produced, so regenerating
# a pickle stays consistent with the checkpoints already trained against it.
# Pass --emit-phase / --no-emit-phase to override, but note that flipping
# FD003 to True changes its training objective and its results are then not
# comparable with the existing FD003 checkpoints.
EMIT_PHASE = {'FD001': True, 'FD003': False}


def read_raw(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name), sep=r'\s+', header=None, names=COLS)


def expanding_baseline(df, cols, max_window=BASELINE_WINDOW):
    """Per-row baseline from that engine's own past only.

    At cycle t the baseline is the mean of cycles 1..min(t, max_window):
    cycle 1 sees a single cycle, cycle 15 the mean of 1..15, and from cycle
    max_window on it settles to the mean of 1..max_window. Never any future
    cycle, and never another engine's data.
    """
    out_frames = []
    for uid, g in df.groupby('unit', sort=False):
        g = g.sort_values('cycle').reset_index(drop=True)
        vals = g[cols].values
        n = len(vals)
        cum_sum = np.cumsum(vals, axis=0)
        idx = np.arange(1, n + 1)
        running_mean = np.empty_like(vals, dtype=float)
        cap_point = min(max_window, n)
        running_mean[:cap_point] = cum_sum[:cap_point] / idx[:cap_point, None]
        if n > max_window:
            running_mean[max_window:] = cum_sum[max_window - 1] / max_window
        out = pd.DataFrame(running_mean, columns=[f'{c}_base_raw' for c in cols])
        out['unit'] = uid
        out['cycle'] = g['cycle'].values
        out['baseline_n_obs'] = np.minimum(idx, max_window)
        out_frames.append(out)
    return pd.concat(out_frames, ignore_index=True)


def smooth_weight(rul_capped, rul_cap=RUL_CAP, w_max=W_MAX, gamma=GAMMA):
    """Continuous weight rising from 1 far from failure to 1 + w_max at
    failure. No changepoint is assumed: the weight is a smooth function of
    how far into the capped degradation range the row sits.

    Because it is a function of the CAPPED RUL, every row with true RUL above
    the cap receives the same minimum weight. That is deliberate -- those rows
    are all equally uninformative about imminent failure -- but it does mean
    this is capped-RUL weighting, not true-distance-to-failure weighting.
    """
    frac = np.clip(1.0 - (rul_capped / rul_cap), 0.0, 1.0)
    return 1.0 + w_max * (frac ** gamma)


def build(subset, data_dir, output_dir, emit_phase=None):
    emit_phase = EMIT_PHASE.get(subset, True) if emit_phase is None else emit_phase
    print("=" * 70)
    print(f"{subset} PREPROCESSING -- causal, leak-free")
    print(f"RUL_CAP={RUL_CAP}  BASELINE_WINDOW={BASELINE_WINDOW}  sensors={len(SELECTED)}")
    print("=" * 70)

    # ── 1. load ──
    tr_raw = read_raw(data_dir, f'train_{subset}.txt')
    te_raw = read_raw(data_dir, f'test_{subset}.txt')
    mc = tr_raw.groupby('unit')['cycle'].max().rename('_mc')
    tr_raw = tr_raw.join(mc, on='unit')
    tr_raw['rul_true'] = tr_raw['_mc'] - tr_raw['cycle']
    tr_raw.drop(columns='_mc', inplace=True)
    print(f"\n  train: {tr_raw['unit'].nunique()} engines | {len(tr_raw):,} rows")
    print(f"  test : {te_raw['unit'].nunique()} engines | {len(te_raw):,} rows")

    # Confirm the single-operating-condition assumption from the data rather
    # than from the dataset documentation, since skipping regime
    # normalisation depends on it.
    op_std = tr_raw[['op1', 'op2', 'op3']].std()
    single_condition = op_std.max() < 0.01
    print(f"  operating-condition std: {op_std.values.round(6)} "
          f"{'-> single condition confirmed' if single_condition else '-> NON-TRIVIAL, regime normalisation needed'}")

    # ── 2. trajectory: global z-score, fitted on train only ──
    mu_g = tr_raw[SELECTED].mean().values
    sig_g = np.maximum(tr_raw[SELECTED].std().values, EPS)
    tr_traj = tr_raw[['unit', 'cycle']].copy()
    te_traj = te_raw[['unit', 'cycle']].copy()
    tr_traj[SELECTED] = (tr_raw[SELECTED].values - mu_g) / sig_g
    te_traj[SELECTED] = (te_raw[SELECTED].values - mu_g) / sig_g

    # ── 3. causal expanding baseline ──
    tr_base_raw = expanding_baseline(tr_raw, SELECTED)
    te_base_raw = expanding_baseline(te_raw, SELECTED)
    base_raw_cols = [f'{c}_base_raw' for c in SELECTED]
    base_mu = tr_base_raw[base_raw_cols].mean().values
    base_sig = np.maximum(tr_base_raw[base_raw_cols].std().values, EPS)
    base_cols = [f'{s}_base' for s in SELECTED]
    tr_base = tr_base_raw[['unit', 'cycle', 'baseline_n_obs']].copy()
    te_base = te_base_raw[['unit', 'cycle', 'baseline_n_obs']].copy()
    tr_base[base_cols] = (tr_base_raw[base_raw_cols].values - base_mu) / base_sig
    te_base[base_cols] = (te_base_raw[base_raw_cols].values - base_mu) / base_sig
    print(f"  rows on a weak (<{BASELINE_WINDOW}-cycle) baseline: "
          f"train={(tr_base['baseline_n_obs'] < BASELINE_WINDOW).sum()}, "
          f"test={(te_base['baseline_n_obs'] < BASELINE_WINDOW).sum()} (early rows, expected)")

    # ── 4. label ──
    tr_traj = tr_traj.merge(tr_raw[['unit', 'cycle', 'rul_true']], on=['unit', 'cycle'])
    tr_traj['rul'] = np.minimum(tr_traj['rul_true'], RUL_CAP)
    tr_traj['sample_weight'] = smooth_weight(tr_traj['rul'].values)

    # ── 5. assemble ──
    train_feat = tr_traj.merge(tr_base, on=['unit', 'cycle'])
    test_feat = te_traj.merge(te_base, on=['unit', 'cycle'])
    all_feat_cols = SELECTED + base_cols
    for df in (train_feat, test_feat):
        df[all_feat_cols] = df[all_feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Degradation onset: the cycle from which the uncapped RUL would fall
    # below the cap, i.e. where the label stops being flat and starts
    # decreasing. phase marks rows at or after it.
    if emit_phase:
        T_i = train_feat.groupby('unit')['cycle'].transform('max').astype(np.float32)
        train_feat['T_i'] = T_i.values.astype(np.float32)
        train_feat['onset'] = np.maximum(0.0, T_i.values - RUL_CAP).astype(np.float32)
        train_feat['phase'] = (train_feat['cycle'].values >= train_feat['onset'].values).astype(np.float32)
    print(f"  onset/T_i/phase columns emitted: {emit_phase}"
          f"{'' if emit_phase else '  (trainer falls back to onset=0, T_i=len, phase=1)'}")

    output = {
        'trajectory_feature_names': {subset: SELECTED},
        'baseline_feature_names': {subset: base_cols},
        'feature_data': {subset: {'train': train_feat, 'test': test_feat}},
        'weighting_config': {'w_max': W_MAX, 'gamma': GAMMA, 'rul_cap': RUL_CAP},
        'asymmetric_loss_config': {'h1': 13.0, 'h2': 10.0},
        'normalization_stats': {subset: {
            'trajectory_mu': mu_g, 'trajectory_sig': sig_g,
            'baseline_mu': base_mu, 'baseline_sig': base_sig,
            'sensor_order': SELECTED}},
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'output_{subset}_{VERSION.get(subset, "v1")}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\n  saved {os.path.basename(out_path)} ({os.path.getsize(out_path) // 1024} KB)")

    verify(train_feat, test_feat, all_feat_cols, single_condition)
    return output


def verify(train_feat, test_feat, all_feat_cols, single_condition):
    """Checks the properties the pipeline claims, rather than assuming them."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    issues = []

    for name, df in [('train', train_feat), ('test', test_feat)]:
        nan_ct = int(df[all_feat_cols].isna().sum().sum())
        inf_ct = int(np.isinf(df[all_feat_cols].values).sum())
        print(f"  [{name}] NaN={nan_ct}  Inf={inf_ct}  {'OK' if not (nan_ct or inf_ct) else 'FAIL'}")
        if nan_ct or inf_ct:
            issues.append(f'{name} has NaN/Inf')

    rmax = train_feat['rul'].max()
    print(f"  RUL cap respected: max={rmax:.0f} (cap={RUL_CAP})  {'OK' if rmax <= RUL_CAP else 'FAIL'}")
    if rmax > RUL_CAP:
        issues.append('RUL cap violated')

    # The baseline at cycle t must be built from exactly min(t, WINDOW)
    # cycles, and never from more cycles than have occurred.
    for name, df in [('train', train_feat), ('test', test_feat)]:
        expected = np.minimum(df['cycle'].values, BASELINE_WINDOW)
        exact = bool((df['baseline_n_obs'].values == expected).all())
        no_future = bool((df['baseline_n_obs'].values <= df['cycle'].values).all())
        print(f"  [{name}] baseline causality: exact={exact} no_future={no_future}  "
              f"{'OK' if exact and no_future else 'FAIL'}")
        if not (exact and no_future):
            issues.append(f'{name} baseline causality violated')

    # sample_weight: check the FUNCTION's mathematical property over the whole
    # range, not just that the realised column happens to come out sorted.
    probe = smooth_weight(np.linspace(0, RUL_CAP, 500))
    monotonic = bool(np.all(np.diff(probe) <= 1e-12))
    print(f"  sample_weight non-increasing in rul: {monotonic}  {'OK' if monotonic else 'FAIL'}")
    if not monotonic:
        issues.append('sample_weight not monotonic')

    wmin = float(train_feat['sample_weight'].min())
    wmax = float(train_feat['sample_weight'].max())
    w_ok = bool(np.isclose(wmin, 1.0, atol=0.05) and np.isclose(wmax, 1 + W_MAX, atol=0.05))
    print(f"  sample_weight range: [{wmin:.2f}, {wmax:.2f}] "
          f"expected [1.00, {1 + W_MAX:.2f}]  {'OK' if w_ok else 'FAIL'}")
    if not w_ok:
        issues.append('sample_weight range off expected bounds')

    if not single_condition:
        issues.append('operating conditions vary; regime normalisation was skipped')

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if not issues else f"{len(issues)} ISSUE(S): {issues}")
    print("=" * 70)
    return issues
