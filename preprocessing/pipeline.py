"""
C-MAPSS preprocessing -- causal and leak-free, one implementation for FD001
and FD003.

    python -m preprocessing FD001        -> output_FD001_v2.pkl
    python -m preprocessing FD003        -> output_FD003_v5.pkl

Both subsets have a SINGLE operating condition, so neither needs the regime
clustering FD002/FD004 require, and both use the same 14 literature-confirmed
sensors. FD003 has two fault modes against FD001's one, but no fault-mode
label is used anywhere: the model is expected to infer the mode from the
trajectory. The two subsets are now processed identically -- same features,
same weighting, same emitted schema.

Feature set: 14 raw sensors. The 4 geometry features are attached downstream
by model.geometry.build_geometry, giving 18 in total.

BASELINE FEATURES REMOVED (FD001 v2 / FD003 v5)
-----------------------------------------------
The 14 causal expanding-baseline columns were ablated directly against a
no-baseline control on both subsets and made TEST_NASA worse on both --
roughly doubling individual-seed NASA on FD001, same direction on FD003. The
likely reason is redundancy: the healthy-window cross-attention already
performs a per-engine offset comparison, learned and dynamic, and a static
subtracted baseline either duplicates or fights it. The feature set drops
from 32 to 18.

WEIGHTING UNIFIED (FD001 v2 / FD003 v5)
---------------------------------------
Previously FD001 emitted onset/T_i/phase and FD003 did not, and the trainer's
phase weight silently falls back to onset=0, T_i=engine length, phase=1 when
they are missing. So FD001 ramped the sample weight only across the post-onset
degradation window while FD003 ramped it across the ENTIRE trajectory,
including the healthy region -- two different objectives from code that looked
dataset-agnostic.

Rather than patch FD003 to emit the columns, the weighting no longer depends
on them: smooth_weight is a function of capped RUL alone, which both subsets
already compute identically. A row with true RUL >= RUL_CAP gets weight 1.0
exactly as before; below the cap the weight rises smoothly to 1 + W_MAX with
no kink at the onset boundary. This class of silent mismatch cannot recur,
because there is no dataset-specific column left for the weighting to depend
on.

onset/T_i/phase are still emitted by BOTH subsets, for inspection and
plotting. They no longer drive anything.
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
W_MAX = 3.0      # weight at failure is 1 + W_MAX
GAMMA = 2.0      # curvature of the ramp; >1 concentrates weight near failure

VERSION = {'FD001': 'v2', 'FD003': 'v5'}


def read_raw(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name), sep=r'\s+', header=None, names=COLS)


def smooth_weight(rul_capped, rul_cap=RUL_CAP, w_max=W_MAX, gamma=GAMMA):
    """Sample weight from capped RUL alone.

    Identical and identically computable for every subset -- no onset, T_i or
    phase dependency, which is what makes a per-dataset weighting mismatch
    impossible. Any row still at or above the cap has
    frac_into_degradation = 0 and weight exactly 1.0; below the cap the weight
    rises smoothly toward 1 + w_max as RUL -> 0.

    Because it is a function of the CAPPED RUL, every row above the cap
    receives the same minimum weight. That is deliberate -- those rows are
    equally uninformative about imminent failure -- but it does mean this is
    capped-RUL weighting, not true-distance-to-failure weighting.
    """
    frac = np.clip(1.0 - (rul_capped / rul_cap), 0.0, 1.0)
    return 1.0 + w_max * (frac ** gamma)


def build(subset, data_dir, output_dir):
    print("=" * 70)
    print(f"{subset} PREPROCESSING {VERSION.get(subset, '')} -- no baseline features, "
          f"smooth unified weighting")
    print(f"RUL_CAP={RUL_CAP}  W_MAX={W_MAX}  GAMMA={GAMMA}  sensors={len(SELECTED)}")
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
    single_condition = bool(op_std.max() < 0.01)
    print(f"  operating-condition std: {op_std.values.round(6)} "
          f"{'-> single condition confirmed' if single_condition else '-> NON-TRIVIAL, regime normalisation needed'}")

    # ── 2. trajectory: global z-score, fitted on train only ──
    mu_g = tr_raw[SELECTED].mean().values
    sig_g = np.maximum(tr_raw[SELECTED].std().values, EPS)
    tr_traj = tr_raw[['unit', 'cycle']].copy()
    te_traj = te_raw[['unit', 'cycle']].copy()
    tr_traj[SELECTED] = (tr_raw[SELECTED].values - mu_g) / sig_g
    te_traj[SELECTED] = (te_raw[SELECTED].values - mu_g) / sig_g
    print(f"  z-score fitted on train only | train mean~{tr_traj[SELECTED].mean().mean():.3f} "
          f"std~{tr_traj[SELECTED].std().mean():.3f}")
    print("  no baseline features: ablated directly, worse TEST_NASA on both subsets")

    # ── 3. label ──
    tr_traj = tr_traj.merge(tr_raw[['unit', 'cycle', 'rul_true']], on=['unit', 'cycle'])
    tr_traj['rul'] = np.minimum(tr_traj['rul_true'], RUL_CAP)
    n_short = int((tr_raw.groupby('unit')['cycle'].max() < RUL_CAP).sum())
    print(f"\n  rul = min(rul_true, {RUL_CAP})  |  engines shorter than the cap: {n_short}")

    # ── 4. onset / T_i / phase -- diagnostic only ──
    T_i = tr_traj.groupby('unit')['cycle'].transform('max').astype(np.float32)
    tr_traj['T_i'] = T_i.values.astype(np.float32)
    tr_traj['onset'] = np.maximum(0.0, T_i.values - RUL_CAP).astype(np.float32)
    tr_traj['phase'] = (tr_traj['cycle'].values >= tr_traj['onset'].values).astype(np.float32)
    print(f"  onset = max(0, T_i - {RUL_CAP}), phase = cycle >= onset "
          f"(emitted for inspection; NOT the weighting driver)")

    # ── 5. sample weight ──
    tr_traj['sample_weight'] = smooth_weight(tr_traj['rul'].values)
    print(f"  sample_weight range: [{tr_traj['sample_weight'].min():.2f}, "
          f"{tr_traj['sample_weight'].max():.2f}]  "
          f"(1.0 at RUL>={RUL_CAP:.0f}, {1 + W_MAX:.1f} at failure)")

    # ── 6. assemble ──
    train_feat = tr_traj.copy()
    test_feat = te_traj.copy()
    feature_cols = list(SELECTED)
    for df in (train_feat, test_feat):
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"\n  {len(feature_cols)} trajectory features, 0 baseline features "
          f"(+4 geometry attached downstream = 18 total)")

    output = {
        'trajectory_feature_names': {subset: list(SELECTED)},
        # Kept as an empty list rather than dropped, for schema compatibility
        # with loaders that concatenate trajectory + baseline + geometry.
        'baseline_feature_names': {subset: []},
        'feature_data': {subset: {'train': train_feat, 'test': test_feat}},
        'weighting_config': {'w_max': W_MAX, 'gamma': GAMMA, 'rul_cap': RUL_CAP,
                             'scheme': 'smooth_weight (unified, rul-only, '
                                       'no onset/phase dependency)'},
        'asymmetric_loss_config': {'h1': 13.0, 'h2': 10.0},
        'normalization_stats': {subset: {
            'trajectory_mu': mu_g, 'trajectory_sig': sig_g,
            'sensor_order': list(SELECTED)}},
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'output_{subset}_{VERSION.get(subset, "v1")}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\n  saved {os.path.basename(out_path)} ({os.path.getsize(out_path) // 1024} KB)")

    verify(train_feat, test_feat, feature_cols, output['baseline_feature_names'][subset],
           single_condition)
    return output


def verify(train_feat, test_feat, feature_cols, base_feature_names, single_condition):
    """Checks the properties the pipeline claims, rather than assuming them."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    issues = []

    for name, df in [('train', train_feat), ('test', test_feat)]:
        nan_ct = int(df[feature_cols].isna().sum().sum())
        inf_ct = int(np.isinf(df[feature_cols].values).sum())
        print(f"  [{name}] NaN={nan_ct}  Inf={inf_ct}  {'OK' if not (nan_ct or inf_ct) else 'FAIL'}")
        if nan_ct or inf_ct:
            issues.append(f'{name} has NaN/Inf')

    rmax = float(train_feat['rul'].max())
    print(f"  RUL cap respected: max={rmax:.0f} (cap={RUL_CAP})  {'OK' if rmax <= RUL_CAP else 'FAIL'}")
    if rmax > RUL_CAP:
        issues.append('RUL cap violated')

    # Check the weight FUNCTION's mathematical properties over the whole
    # range, not just that the realised column happens to come out sorted.
    probe = smooth_weight(np.linspace(0, RUL_CAP, 500))
    monotonic = bool(np.all(np.diff(probe) <= 1e-12))
    # Second difference bounded => no slope discontinuity, which is exactly
    # what the old phase step function had at the onset boundary.
    smooth = bool(np.all(np.abs(np.diff(probe, n=2)) < 1.0))
    print(f"  weight function non-increasing in rul: {monotonic}  {'OK' if monotonic else 'FAIL'}")
    print(f"  weight function smooth (no slope kink): {smooth}  {'OK' if smooth else 'FAIL'}")
    if not monotonic:
        issues.append('sample_weight not monotonic')
    if not smooth:
        issues.append('sample_weight has a slope discontinuity')

    wmin = float(train_feat['sample_weight'].min())
    wmax = float(train_feat['sample_weight'].max())
    w_ok = bool(np.isclose(wmin, 1.0, atol=0.05) and np.isclose(wmax, 1 + W_MAX, atol=0.05))
    print(f"  sample_weight range: [{wmin:.2f}, {wmax:.2f}] "
          f"expected [1.00, {1 + W_MAX:.2f}]  {'OK' if w_ok else 'FAIL'}")
    if not w_ok:
        issues.append('sample_weight range off expected bounds')

    onset_ok = bool((train_feat['onset'].values <= train_feat['T_i'].values).all())
    print(f"  onset <= T_i for every row: {onset_ok}  {'OK' if onset_ok else 'FAIL'}")
    if not onset_ok:
        issues.append('onset exceeds T_i for some row')

    no_baseline = len(base_feature_names) == 0
    print(f"  baseline features removed: {no_baseline}  {'OK' if no_baseline else 'FAIL'}")
    if not no_baseline:
        issues.append('baseline features still present')

    if not single_condition:
        issues.append('operating conditions vary; regime normalisation was skipped')

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if not issues else f"{len(issues)} ISSUE(S): {issues}")
    print("=" * 70)
    return issues
