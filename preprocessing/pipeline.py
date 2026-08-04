"""
C-MAPSS preprocessing -- causal and leak-free, one implementation for FD001
and FD003.

    python -m preprocessing FD001        -> output_FD001_final.pkl
    python -m preprocessing FD003        -> output_FD003_final.pkl

Both subsets have a SINGLE operating condition, so neither needs the regime
clustering FD002/FD004 require, and both use the same 14 literature-confirmed
sensors. FD003 has two fault modes against FD001's one, but no fault-mode
label is used anywhere: the model is expected to infer the mode from the
trajectory. The two subsets are now processed identically -- same features,
same weighting, same emitted schema.

Feature set: 14 raw sensors. The 4 geometry features are attached downstream
by model.geometry.build_geometry, giving 18 in total.

BASELINE FEATURES REMOVED
-------------------------
The 14 causal expanding-baseline columns were ablated directly against a
no-baseline control on both subsets and made TEST_NASA worse on both --
roughly doubling individual-seed NASA on FD001, same direction on FD003. The
likely reason is redundancy: the healthy-window cross-attention already
performs a per-engine offset comparison, learned and dynamic, and a static
subtracted baseline either duplicates or fights it. The feature set drops
from 32 to 18.

SAMPLE WEIGHTING REMOVED
------------------------
Weighting went through two rounds. The first fixed a real bug: FD001 emitted
onset/T_i/phase and FD003 did not, the trainer silently substituted onset=0,
phase=1 when they were absent, and the two subsets ended up trained on
different objectives from code that looked dataset-agnostic. The fix made the
weight a function of capped RUL alone.

The second round ablated that unified weight against a plain UNWEIGHTED
asymmetric Huber, and unweighted won -- as did unweighted against
asym_plus_nasa_surrogate. So there is no weight to emit: no sample_weight
column, and no onset/T_i/phase either, since those only ever existed to feed
a phase weight that no longer exists.

What is left is exactly the two things the model consumes: 14 z-scored
trajectory features and the capped RUL label. Verification asserts that
schema is closed, so a dead column cannot creep back in and shift a
downstream feature count.
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

# One label for both subsets: they are now processed identically, so a
# per-subset version number said nothing except which historical revision each
# had reached. 'final' is the settled state -- no baseline features, no
# weighting, nothing left that an ablation still has an opinion about.
VERSION = {'FD001': 'final', 'FD003': 'final'}


def read_raw(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name), sep=r'\s+', header=None, names=COLS)


def build(subset, data_dir, output_dir):
    print("=" * 70)
    print(f"{subset} PREPROCESSING {VERSION.get(subset, '')} -- trajectory + RUL only, "
          f"no baseline features, no sample weighting")
    print(f"RUL_CAP={RUL_CAP}  sensors={len(SELECTED)}")
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

    print("  no sample_weight: the unified RUL-only weight lost to plain "
          "unweighted asym_huber in the loss ablation")

    # ── 4. assemble ──
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
        'weighting_config': {'rul_cap': RUL_CAP,
                             'scheme': 'none -- unweighted asym_huber'},
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

    # The emitted schema is now closed: anything beyond these columns means a
    # dead column has crept back in and downstream feature counts can drift.
    expected = {'unit', 'cycle', 'rul_true', 'rul'} | set(SELECTED)
    extra = set(train_feat.columns) - expected
    print(f"  emits trajectory + RUL only: {not extra}  {'OK' if not extra else f'FAIL {extra}'}")
    if extra:
        issues.append(f'unexpected columns emitted: {sorted(extra)}')

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
