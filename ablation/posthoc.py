"""
Post-hoc statistics over saved results. No torch, no checkpoints, no GPU.

    python -m ablation.posthoc
    python -m ablation.posthoc --ensemble mean --datasets FD001 FD004

Reads what the earlier stages wrote -- per-seed results and the combined
ensemble predictions -- and runs the tests a write-up needs, in three parts:

  A  direct training, per dataset
     A1  does the reported ensemble beat the plain mean, paired per ENGINE
     A2  seed spread: mean, sd, coefficient of variation, CI of the mean
     A3  cross-dataset consistency
     A4  bootstrap CI on the reported NASA and RMSE
     A5  are the errors normal (Shapiro-Wilk)

  B  the transfer arm, the same tests where they apply

  C  transfer vs direct training on the shared target -- the headline claim,
     paired per engine, which is the only place both pipelines meet

Two things this module refuses to pretend:

  NASA is a SUM over engines. Its bootstrap CI is computed by resampling
  engines WITH replacement at the original count, so the sum stays on the
  same scale; it is still not comparable between datasets of different size,
  and the printout says which n it came from.

  A3 is reported as a DESCRIPTION, not a test. Friedman needs blocks that
  are matched units, and seed 53 on FD001 is not the same unit as seed 53 on
  FD004 -- different engines, different splits, a shared RNG value and
  nothing more. Running it anyway would produce a p-value that looks like
  evidence about datasets and is really about an arbitrary pairing, so what
  is printed is each dataset's spread side by side.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from model.config import DATA_ROOT, OUT_ROOT, RUL_CAP, DATASETS
from model.data import raw_path
from model.metrics import nasa_score, rmse

from .ensembling import run_ensembling_suite
from .stats import effect_size_label, normality_p, paired_comparison

SETTLED_CONFIG = 'sensor_branch_linear_far'
TRANSFER_ARM = 'FD002_to_FD004_frozen'
TRANSFER_TARGET = 'FD004'
# The technique the write-up reports. Everything in section A1 is measured
# against the plain mean, so this is the claim being tested rather than an
# assumption being made.
REPORTED_ENSEMBLE = 'val_weighted'
N_BOOT = 10000
RNG_SEED = 0


def load_members(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_rul_true(subset, data_root):
    rul_file = raw_path(subset, data_root)
    if not os.path.exists(rul_file):
        return None
    return np.minimum(
        pd.read_csv(rul_file, header=None, names=['rul']).values.ravel(), RUL_CAP)


def ensembles_from(members, rul_true):
    """Rebuilds the combined predictions from the per-seed members.

    Recomputed rather than read from the .npz so this module works on any run
    that saved its members, and so the ensemble can be re-derived if the
    suite changes.
    """
    _, _, combined = run_ensembling_suite(members, rul_true)
    return {name: preds for name, preds in combined.items() if preds is not None}


def bootstrap_metric_ci(pred, rul_true, metric_fn, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile CI by resampling ENGINES with replacement.

    The resample keeps the original engine count, which matters for NASA:
    it is a sum, so a smaller resample would shrink it for a reason that has
    nothing to do with the model.
    """
    rng = np.random.default_rng(seed)
    pred, rul_true = np.asarray(pred), np.asarray(rul_true)
    n = len(rul_true)
    vals = np.array([metric_fn(pred[i], rul_true[i])
                     for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(vals.mean()), (float(lo), float(hi))


def test_ensembling_benefit(label, ens, rul_true, reported):
    """The reported technique against the plain mean, paired per engine."""
    if reported not in ens or 'mean' not in ens:
        print(f"  {label}: no '{reported}' or 'mean' among {sorted(ens)} -- skipped")
        return None
    err_reported = np.abs(ens[reported] - rul_true)
    err_mean = np.abs(ens['mean'] - rul_true)
    # Sign: mean minus reported, so positive means the reported technique has
    # the smaller error and wins.
    cmp_ = paired_comparison(err_mean, err_reported)
    if not cmp_['ci_excludes_zero']:
        verdict = 'within noise'
    elif cmp_['mean_diff'] > 0:
        verdict = f'{reported} wins'
    else:
        verdict = 'mean wins'
    print(f"  {label}: |error| {reported} vs mean over {cmp_['n']} engines -- {verdict}")
    print(f"    mean diff={cmp_['mean_diff']:+.3f}  paired-t p={cmp_['p_ttest']:.4g}  "
          f"wilcoxon p={cmp_['p_wilcoxon']:.4g}  d={cmp_['cohens_d']:+.3f} "
          f"({effect_size_label(cmp_['cohens_d'])})  "
          f"CI=({cmp_['bootstrap_ci95'][0]:+.3f}, {cmp_['bootstrap_ci95'][1]:+.3f})")
    return cmp_


def test_seed_spread(label, members):
    """How much the result moves with the seed alone.

    The coefficient of variation is the number to quote: an sd of 30 means
    something different against a NASA of 300 than against 3000, and NASA
    varies by an order of magnitude between subsets.
    """
    print(f"  {label} ({len(members)} seeds):")
    out = {}
    rng = np.random.default_rng(RNG_SEED)
    for metric in ('nasa', 'rmse'):
        vals = np.array([m[metric] for m in members], dtype=np.float64)
        mean, sd = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        cv = (sd / mean * 100) if mean else float('nan')
        boot = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
        lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
        print(f"    {metric:<5} mean={mean:.3f}  sd={sd:.3f}  CV={cv:.2f}%  "
              f"95% CI of the mean=({lo:.3f}, {hi:.3f})")
        out[metric] = dict(mean=mean, sd=sd, cv=cv, ci=(lo, hi))
    return out


def test_bootstrap_ci(label, ens, rul_true, reported):
    if reported not in ens:
        print(f"  {label}: no '{reported}' -- skipped")
        return None
    nasa_mean, nasa_ci = bootstrap_metric_ci(ens[reported], rul_true, nasa_score)
    rmse_mean, rmse_ci = bootstrap_metric_ci(ens[reported], rul_true, rmse)
    print(f"  {label} ({len(rul_true)} engines): "
          f"NASA={nasa_mean:.2f} CI=({nasa_ci[0]:.2f}, {nasa_ci[1]:.2f})   "
          f"RMSE={rmse_mean:.3f} CI=({rmse_ci[0]:.3f}, {rmse_ci[1]:.3f})")
    return dict(nasa=(nasa_mean, nasa_ci), rmse=(rmse_mean, rmse_ci))


def test_error_normality(label, ens, rul_true, reported):
    """Shapiro-Wilk on the prediction errors.

    Errors here are expected NOT to be normal -- the RUL cap piles the label
    at 125 and the loss is asymmetric on purpose -- so a rejection is
    information about the data, not a fault. It is reported because it says
    which summary statistics are safe: a mean error is misleading on a skewed
    distribution in the way that matters here.
    """
    if reported not in ens:
        print(f"  {label}: no '{reported}' -- skipped")
        return None
    errs = ens[reported] - rul_true
    try:
        w, p = scipy_stats.shapiro(errs)
    except ValueError:
        print(f"  {label}: Shapiro-Wilk not computable")
        return None
    verdict = 'errors are not normal' if p < 0.05 else 'consistent with normal'
    skew = float(scipy_stats.skew(errs))
    lean = 'over-prediction' if skew > 0 else 'under-prediction'
    print(f"  {label}: W={w:.4f}  p={p:.4g} -- {verdict};  skew={skew:+.2f} "
          f"(tail toward {lean})")
    return dict(w=float(w), p=float(p), skew=skew)


def test_transfer_vs_direct(arm, transfer_ens, direct_ens, rul_true, target, reported):
    """The headline: transfer against training on the target directly.

    Paired per engine, which is the strongest pairing available -- both sides
    predicted the same engines -- and the only comparison that needs both
    pipelines at once.
    """
    if reported not in transfer_ens or reported not in direct_ens:
        print(f"  {arm}: '{reported}' missing on one side -- skipped")
        return None
    err_transfer = np.abs(transfer_ens[reported] - rul_true)
    err_direct = np.abs(direct_ens[reported] - rul_true)
    # Sign: direct minus transfer, so positive means transfer had the smaller
    # error and won.
    cmp_ = paired_comparison(err_direct, err_transfer)
    if not cmp_['ci_excludes_zero']:
        verdict = (f'within noise -- transfer neither helps nor hurts against '
                   f'training on {target} directly')
    elif cmp_['mean_diff'] > 0:
        verdict = 'transfer wins'
    else:
        verdict = 'direct training wins'
    print(f"  {arm} vs direct training on {target}, over {cmp_['n']} engines -- {verdict}")
    print(f"    |error| diff (direct - transfer)={cmp_['mean_diff']:+.3f}  "
          f"paired-t p={cmp_['p_ttest']:.4g}  wilcoxon p={cmp_['p_wilcoxon']:.4g}  "
          f"d={cmp_['cohens_d']:+.3f} ({effect_size_label(cmp_['cohens_d'])})  "
          f"CI=({cmp_['bootstrap_ci95'][0]:+.3f}, {cmp_['bootstrap_ci95'][1]:+.3f})")
    shapiro = normality_p(err_direct, err_transfer)
    if shapiro is not None:
        print(f"    shapiro p={shapiro:.3f} on the paired differences")
    return cmp_


def _header(text):
    print(f"\n{'=' * 90}\n{text}\n{'=' * 90}")


def run(subsets, config=SETTLED_CONFIG, reported=REPORTED_ENSEMBLE,
        data_root=None, out_root=None):
    data_root = data_root or DATA_ROOT
    out_root = out_root or OUT_ROOT

    direct = {}
    for subset in subsets:
        members = load_members(os.path.join(out_root, subset, f'{config}_results.pkl'))
        rul_true = load_rul_true(subset, data_root)
        if members and rul_true is not None:
            direct[subset] = dict(members=members, rul_true=rul_true,
                                  ens=ensembles_from(members, rul_true))
        else:
            print(f"  {subset}: no saved results for {config} -- skipped")

    print("\n" + "#" * 90)
    print(f"# A -- DIRECT TRAINING, {config}")
    print("#" * 90)

    _header(f"A1  is '{reported}' better than the plain mean? (paired per engine)")
    for subset, d in direct.items():
        test_ensembling_benefit(subset, d['ens'], d['rul_true'], reported)

    _header("A2  seed spread")
    for subset, d in direct.items():
        test_seed_spread(subset, d['members'])

    _header("A3  cross-dataset spread (a description, NOT a test)")
    print("  Seeds are not matched units across datasets -- seed 53 on FD001 and on")
    print("  FD004 share an RNG value and nothing else -- so a repeated-measures test")
    print("  here would put a p-value on an arbitrary pairing. Spreads, side by side:")
    for subset, d in direct.items():
        nasa = np.array([m['nasa'] for m in d['members']])
        rm = np.array([m['rmse'] for m in d['members']])
        print(f"    {subset:<7} n_engines={len(d['rul_true']):<4} "
              f"NASA {nasa.mean():>8.2f} +/- {nasa.std():<7.2f} "
              f"RMSE {rm.mean():>6.3f} +/- {rm.std():.3f}")
    print("  NASA is a sum, so its column tracks engine count as well as quality;")
    print("  RMSE is the column to compare across datasets.")

    _header(f"A4  bootstrap 95% CI on the reported '{reported}'")
    for subset, d in direct.items():
        test_bootstrap_ci(subset, d['ens'], d['rul_true'], reported)

    _header("A5  error normality")
    for subset, d in direct.items():
        test_error_normality(subset, d['ens'], d['rul_true'], reported)

    # ---- B and C: the transfer arm ----
    arm_path = os.path.join(out_root, 'transfer', TRANSFER_ARM, 'results.pkl')
    arm_members = load_members(arm_path)
    if not arm_members:
        print(f"\n  no transfer results at {arm_path} -- sections B and C skipped. Run:\n"
              f"    python -m ablation.transfer")
        return direct

    target_rul = load_rul_true(TRANSFER_TARGET, data_root)
    arm_ens = ensembles_from(arm_members, target_rul)

    print("\n" + "#" * 90)
    print(f"# B -- TRANSFER ARM, {TRANSFER_ARM}")
    print("#" * 90)
    _header(f"B1  is '{reported}' better than the plain mean?")
    test_ensembling_benefit(TRANSFER_ARM, arm_ens, target_rul, reported)
    _header("B2  seed spread")
    test_seed_spread(TRANSFER_ARM, arm_members)
    _header(f"B3  bootstrap 95% CI on '{reported}'")
    test_bootstrap_ci(TRANSFER_ARM, arm_ens, target_rul, reported)
    _header("B4  error normality")
    test_error_normality(TRANSFER_ARM, arm_ens, target_rul, reported)

    print("\n" + "#" * 90)
    print(f"# C -- TRANSFER vs DIRECT TRAINING ON {TRANSFER_TARGET}")
    print("#" * 90 + "\n")
    if TRANSFER_TARGET not in direct:
        print(f"  no direct-training results for {TRANSFER_TARGET} -- the transfer numbers")
        print(f"  above say what transfer scores, not whether it beat the alternative. Run:")
        print(f"    python -m ablation.train --datasets {TRANSFER_TARGET} --configs {config}")
    else:
        test_transfer_vs_direct(TRANSFER_ARM, arm_ens, direct[TRANSFER_TARGET]['ens'],
                                target_rul, TRANSFER_TARGET, reported)
    return direct


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--config', default=SETTLED_CONFIG)
    ap.add_argument('--ensemble', default=REPORTED_ENSEMBLE,
                    help='the technique the write-up reports (default: %(default)s)')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--out-root', default=OUT_ROOT)
    args = ap.parse_args()

    print("=" * 90)
    print(f"POST-HOC STATISTICS -- {args.config}, reporting '{args.ensemble}'")
    print(f"datasets={args.datasets}")
    print("=" * 90)
    if not run(args.datasets, args.config, args.ensemble, args.data_root, args.out_root):
        print("\nNothing loaded. Train and evaluate first:\n    python -m ablation.train")


if __name__ == '__main__':
    main()
