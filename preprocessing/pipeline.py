"""
C-MAPSS preprocessing -- causal and leak-free, one implementation for all four
subsets.

    python -m preprocessing FD001        -> output_FD001_final.pkl
    python -m preprocessing FD002        -> output_FD002_final.pkl
    python -m preprocessing FD003        -> output_FD003_final.pkl
    python -m preprocessing FD004        -> output_FD004_final.pkl

Every subset emits the same schema and the same 14 literature-confirmed
sensors. The only thing that differs is how the sensors are normalised, and
that follows from the operating conditions:

  FD001, FD003   a SINGLE operating condition, so one global z-score fitted
                 on train is enough.
  FD002, FD004   SIX discrete operating conditions with real variance -- an
                 altitude / Mach / throttle-resolver-angle combination the
                 engine is being flown at. A sensor reading means something
                 different in each, so a single z-score would leave regime
                 changes looking like degradation. Each row is normalised
                 against its own regime's train statistics instead.

The regime clustering is NOT the fault-mode discovery that was tried and
dropped. That clustered ENGINES by degradation pattern, inferring a latent
label the data does not carry; this clusters ROWS by op1/op2/op3, three
measured columns with six known discrete settings. One is a guess about the
failure; the other is reading the flight condition off the instrument.

Fault modes are not modelled anywhere. FD003 and FD004 have two against
FD001's and FD002's one, and no fault-mode label is emitted or used -- the
model is expected to infer the mode from the trajectory.

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
OP_COLS = ['op1', 'op2', 'op3']
EPS = 1e-8

# The 14 sensors that vary with degradation under a single operating
# condition; the other 7 are constant or pure noise on these subsets.
SELECTED = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
            's13', 's14', 's15', 's17', 's20', 's21']

RUL_CAP = 125.0

# Subsets flown across several operating conditions, which are normalised per
# regime rather than globally. Declared rather than inferred so the choice is
# reviewable -- and cross-checked against the measured op-condition variance
# in verify(), which fails if the declaration and the data disagree.
REGIME_SUBSETS = {'FD002', 'FD004'}
N_REGIMES = 6         # the six documented C-MAPSS flight conditions
KMEANS_SEED = 0
OP_STD_THRESHOLD = 0.01

# One label for every subset: they are processed identically apart from the
# normalisation, so a per-subset version number said nothing except which
# historical revision each had reached. 'final' is the settled state -- no
# baseline features, no weighting, nothing an ablation still has an opinion
# about.
VERSION = {s: 'final' for s in ('FD001', 'FD002', 'FD003', 'FD004')}


def read_raw(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name), sep=r'\s+', header=None, names=COLS)


def normalise_global(tr_raw, te_raw):
    """One z-score for every row, fitted on train. For the single-condition
    subsets, where op1/op2/op3 barely move."""
    mu = tr_raw[SELECTED].mean().values
    sig = np.maximum(tr_raw[SELECTED].std().values, EPS)
    tr_vals = (tr_raw[SELECTED].values - mu) / sig
    te_vals = (te_raw[SELECTED].values - mu) / sig
    stats = {'scheme': 'global', 'trajectory_mu': mu, 'trajectory_sig': sig,
             'sensor_order': list(SELECTED)}
    return tr_vals, te_vals, stats, None


def normalise_by_regime(tr_raw, te_raw, n_regimes=N_REGIMES, seed=KMEANS_SEED):
    """A z-score per operating regime, everything fitted on train.

    KMeans over the three standardised op columns recovers the regimes; the
    sensors are then z-scored against the statistics of the regime the row was
    flown in. Test rows are ASSIGNED to the nearest train-fit centre and
    normalised with that regime's TRAIN statistics -- no test row contributes
    to any statistic, so the split stays clean.

    Returns the per-row train/test regime labels too, since verification has
    to check that every regime was actually populated: an empty train regime
    would leave its centre with no statistics for test rows that land there.
    """
    op_scaler = StandardScaler().fit(tr_raw[OP_COLS].values)
    tr_op = op_scaler.transform(tr_raw[OP_COLS].values)
    te_op = op_scaler.transform(te_raw[OP_COLS].values)

    kmeans = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10).fit(tr_op)
    tr_regime = kmeans.predict(tr_op)
    te_regime = kmeans.predict(te_op)

    tr_sensors = tr_raw[SELECTED].values
    te_sensors = te_raw[SELECTED].values
    tr_vals = np.zeros_like(tr_sensors, dtype=np.float64)
    te_vals = np.zeros_like(te_sensors, dtype=np.float64)
    regime_mu, regime_sig = {}, {}

    for r in range(n_regimes):
        tr_mask = tr_regime == r
        if not tr_mask.any():
            # No train rows: leave the regime's statistics undefined rather
            # than inventing them. verify() turns this into a failure.
            continue
        mu = tr_sensors[tr_mask].mean(axis=0)
        sig = np.maximum(tr_sensors[tr_mask].std(axis=0), EPS)
        regime_mu[r], regime_sig[r] = mu, sig
        tr_vals[tr_mask] = (tr_sensors[tr_mask] - mu) / sig
        te_mask = te_regime == r
        if te_mask.any():
            te_vals[te_mask] = (te_sensors[te_mask] - mu) / sig

    stats = {'scheme': 'per-regime', 'n_regimes': n_regimes,
             'op_scaler_mean': op_scaler.mean_, 'op_scaler_scale': op_scaler.scale_,
             'kmeans_centers': kmeans.cluster_centers_,
             'regime_mu': regime_mu, 'regime_sig': regime_sig,
             'sensor_order': list(SELECTED)}
    return tr_vals, te_vals, stats, (tr_regime, te_regime)


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

    # Read the operating conditions off the data rather than trusting the
    # dataset documentation: which normalisation is correct depends on them,
    # and verify() fails if this disagrees with what the subset declared.
    op_std = tr_raw[OP_COLS].std()
    single_condition = bool(op_std.max() < OP_STD_THRESHOLD)
    per_regime = subset in REGIME_SUBSETS
    print(f"  operating-condition std: {op_std.values.round(6)} "
          f"{'-> single condition' if single_condition else '-> multiple regimes'}")

    # ── 2. trajectory: z-score fitted on train only ──
    if per_regime:
        print(f"  normalisation: per-regime, KMeans(k={N_REGIMES}) on op1/op2/op3, "
              f"fitted on train only")
        tr_vals, te_vals, norm_stats, regimes = normalise_by_regime(tr_raw, te_raw)
        tr_regime, te_regime = regimes
        tr_counts = pd.Series(tr_regime).value_counts().sort_index().to_dict()
        te_counts = pd.Series(te_regime).value_counts().sort_index().to_dict()
        print(f"  regime rows (train): {tr_counts}")
        print(f"  regime rows (test) : {te_counts}  "
              f"(assigned to the nearest train-fit centre, normalised with its "
              f"train statistics)")
    else:
        print("  normalisation: one global z-score, fitted on train only")
        tr_vals, te_vals, norm_stats, regimes = normalise_global(tr_raw, te_raw)

    tr_traj = tr_raw[['unit', 'cycle']].copy()
    te_traj = te_raw[['unit', 'cycle']].copy()
    tr_traj[SELECTED] = tr_vals
    te_traj[SELECTED] = te_vals
    print(f"  train mean~{tr_traj[SELECTED].values.mean():.3f} "
          f"std~{tr_traj[SELECTED].values.std():.3f}")
    print("  no baseline features: ablated directly, worse TEST_NASA on both "
          "single-condition subsets")

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
        'normalization_stats': {subset: norm_stats},
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'output_{subset}_{VERSION.get(subset, "v1")}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\n  saved {os.path.basename(out_path)} ({os.path.getsize(out_path) // 1024} KB)")

    verify(train_feat, test_feat, feature_cols, output['baseline_feature_names'][subset],
           single_condition, per_regime, regimes)
    return output


def verify(train_feat, test_feat, feature_cols, base_feature_names, single_condition,
           per_regime=False, regimes=None):
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

    # The emitted schema is closed: anything beyond these columns means a dead
    # column has crept back in and downstream feature counts can drift.
    # The same schema on every subset: what differs between them is how the
    # sensors were normalised, not what the model is handed.
    expected = {'unit', 'cycle', 'rul_true', 'rul'} | set(SELECTED)
    extra = set(train_feat.columns) - expected
    print(f"  emits trajectory + RUL only: {not extra}  "
          f"{'OK' if not extra else f'FAIL {extra}'}")
    if extra:
        issues.append(f'unexpected columns emitted: {sorted(extra)}')

    no_baseline = len(base_feature_names) == 0
    print(f"  baseline features removed: {no_baseline}  {'OK' if no_baseline else 'FAIL'}")
    if not no_baseline:
        issues.append('baseline features still present')

    # The declared normalisation has to match what the op columns actually do.
    # Both directions are failures: a multi-regime subset z-scored globally
    # leaves regime changes looking like degradation, and a single-condition
    # subset split into six regimes just partitions noise.
    agrees = per_regime != single_condition
    print(f"  normalisation matches measured op variance: {agrees}  "
          f"{'OK' if agrees else 'FAIL'}")
    if not agrees:
        issues.append('operating conditions vary; regime normalisation was skipped'
                      if single_condition is False else
                      'single operating condition, but regime normalisation was applied')

    if per_regime and regimes is not None:
        tr_regime, te_regime = regimes
        empty = sorted(set(range(N_REGIMES)) - set(np.unique(tr_regime).tolist()))
        print(f"  every regime populated in train: {not empty}  "
              f"{'OK' if not empty else f'FAIL {empty}'}")
        if empty:
            issues.append(f'regimes with no train rows: {empty}')
        # A test row landing in a regime with no train statistics would have
        # been left at zero, which reads as a perfectly average reading.
        stranded = sorted(set(np.unique(te_regime).tolist()) - set(np.unique(tr_regime).tolist()))
        print(f"  no test rows stranded in an unfitted regime: {not stranded}  "
              f"{'OK' if not stranded else f'FAIL {stranded}'}")
        if stranded:
            issues.append(f'test rows in regimes with no train statistics: {stranded}')

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if not issues else f"{len(issues)} ISSUE(S): {issues}")
    print("=" * 70)
    return issues
