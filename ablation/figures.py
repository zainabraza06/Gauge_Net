"""
Report figures for the settled model, drawn from what evaluation already saved.

    python -m ablation.figures
    python -m ablation.figures --datasets FD001 FD004

Reads the ensemble predictions the evaluation stage wrote
(<config>_ensemble_preds.npz) and the ground truth, so it draws exactly the
numbers the tables report -- it never re-runs inference, and a figure can
therefore never disagree with the CSV beside it.

Per dataset, into <out-root>/figures/<dataset>/:

  error_hist        prediction error with a Gaussian fit. The fit is drawn to
                    be argued with: if the errors are skewed, the curve will
                    visibly fail to follow them, which is the useful outcome
                    for a score that punishes late predictions asymmetrically.
  scatter           true vs predicted, against the ideal line. Points above
                    the line are the dangerous direction.
  error_by_true     absolute error against true RUL. RUL is capped at 125, so
                    a model can look good overall and still be weak exactly
                    where it matters, near failure; this is where that shows.
  ensembles         NASA per ensembling technique, best highlighted.

And across datasets, into <out-root>/figures/:

  cross_dataset     NASA and RMSE per dataset. NASA is a SUM, so its bars are
                    only comparable between datasets with equally many test
                    engines -- the engine count is printed on each bar.
  error_spread      absolute error distributions side by side.

No radar chart. Mixing accuracies, R2 and reciprocal errors on one axis makes
an area that looks meaningful and is not: the shape changes with the metric
order, and none of the axes share units.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')          # headless: figures are written, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.config import DATA_ROOT, OUT_ROOT, RUL_CAP, DATASETS
from model.data import raw_path
from model.metrics import all_metrics

SETTLED_CONFIG = 'sensor_branch_linear_far'
DPI = 160
BLUE, ORANGE, GREEN, RED = '#4C72B0', '#DD8452', '#55A868', '#C44E52'


def figure_dir(out_root, subset=None):
    d = os.path.join(out_root, 'figures', subset) if subset else os.path.join(out_root, 'figures')
    os.makedirs(d, exist_ok=True)
    return d


def load_predictions(subset, config, out_root, data_root):
    """(ensemble predictions by technique, ground-truth RUL) for one subset."""
    npz = os.path.join(out_root, subset, f'{config}_ensemble_preds.npz')
    if not os.path.exists(npz):
        return None, None
    preds = dict(np.load(npz))
    rul_file = raw_path(subset, data_root)
    if not os.path.exists(rul_file):
        return None, None
    rul_true = np.minimum(
        pd.read_csv(rul_file, header=None, names=['rul']).values.ravel(), RUL_CAP)
    return preds, rul_true


def plot_error_hist(subset, pred, rul_true, out_root):
    err = np.asarray(pred) - np.asarray(rul_true)
    mu, sigma = float(err.mean()), float(err.std())
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = max(10, min(30, len(err) // 4))
    _, edges, _ = ax.hist(err, bins=bins, density=True, alpha=0.65, color=BLUE,
                          edgecolor='white', label='errors')
    if sigma > 1e-9:
        xs = np.linspace(edges[0], edges[-1], 300)
        ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
                color=ORANGE, lw=2, label=f'Gaussian fit  mu={mu:.2f}, sigma={sigma:.2f}')
    ax.axvline(0, color='black', lw=1, ls='--')
    # The two sides are not equivalent: NASA charges e^(e/10) for predicting
    # too much life against e^(-e/13) for too little, so the right tail is
    # the expensive one and is worth being able to see.
    ax.axvspan(0, max(edges[-1], 1), color=RED, alpha=0.06)
    ax.text(0.98, 0.95, 'over-prediction\n(the costly side)', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=RED)
    ax.set_xlabel('prediction error (predicted - true RUL, cycles)')
    ax.set_ylabel('density')
    ax.set_title(f'{subset}: error distribution, mean-ensemble')
    ax.legend(fontsize=8, loc='upper left')
    return _save(fig, figure_dir(out_root, subset), f'error_hist_{subset}')


def plot_scatter(subset, pred, rul_true, out_root):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(rul_true, pred, s=28, alpha=0.55, color=BLUE, edgecolor='none')
    hi = max(float(np.max(rul_true)), float(np.max(pred))) + 5
    ax.plot([0, hi], [0, hi], '--', color='black', lw=1.2, label='perfect prediction')
    ax.fill_between([0, hi], [0, hi], hi, color=RED, alpha=0.05)
    ax.text(hi * 0.05, hi * 0.93, 'over-predicting RUL', fontsize=8, color=RED)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel('true RUL (cycles)')
    ax.set_ylabel('predicted RUL (cycles)')
    ax.set_title(f'{subset}: predicted vs true, mean-ensemble')
    ax.legend(fontsize=8, loc='lower right')
    return _save(fig, figure_dir(out_root, subset), f'scatter_{subset}')


def plot_error_by_true(subset, pred, rul_true, out_root):
    """Where the error sits along the RUL range.

    The label is capped at 125, so a large share of the test set shares one
    true value; an overall error figure hides whether the model is weak
    exactly where a maintenance decision is made, close to failure.
    """
    err = np.abs(np.asarray(pred) - np.asarray(rul_true))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rul_true, err, s=26, alpha=0.55, color=BLUE, edgecolor='none')
    edges = np.linspace(0, RUL_CAP, 6)
    mids, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rul_true >= lo) & (rul_true < hi if hi < RUL_CAP else rul_true <= hi)
        if sel.sum():
            mids.append((lo + hi) / 2)
            means.append(err[sel].mean())
    if mids:
        ax.plot(mids, means, '-o', color=ORANGE, lw=2, ms=5, label='mean |error| per band')
    ax.set_xlabel('true RUL (cycles)')
    ax.set_ylabel('|prediction error| (cycles)')
    ax.set_title(f'{subset}: error against remaining life')
    ax.legend(fontsize=8)
    return _save(fig, figure_dir(out_root, subset), f'error_by_true_{subset}')


def plot_ensembles(subset, preds, rul_true, out_root):
    names = sorted(preds, key=lambda n: all_metrics(preds[n], rul_true)['nasa'])
    scores = [all_metrics(preds[n], rul_true)['nasa'] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = [ORANGE if n == 'mean' else BLUE for n in names]
    bars = ax.bar(names, scores, color=colours, edgecolor='black', lw=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score, f'{score:.1f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('NASA score (lower is better)')
    ax.set_title(f'{subset}: ensembling techniques, sorted; mean highlighted')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    return _save(fig, figure_dir(out_root, subset), f'ensembles_{subset}')


def plot_cross_dataset(per_subset, out_root):
    subsets = list(per_subset)
    nasa = [per_subset[s]['metrics']['nasa'] for s in subsets]
    rmse = [per_subset[s]['metrics']['rmse'] for s in subsets]
    n_eng = [len(per_subset[s]['rul_true']) for s in subsets]

    x = np.arange(len(subsets))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    b = ax1.bar(x, nasa, color=BLUE, edgecolor='black', lw=0.5)
    for bar, n in zip(b, n_eng):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'n={n}',
                 ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(x, subsets)
    ax1.set_ylabel('NASA score')
    # Saying this on the figure rather than in a caption: the bars invite a
    # comparison the metric does not support.
    ax1.set_title('NASA score -- a SUM, so compare only\nbetween equal engine counts', fontsize=10)

    ax2.bar(x, rmse, color=ORANGE, edgecolor='black', lw=0.5)
    ax2.set_xticks(x, subsets)
    ax2.set_ylabel('RMSE (cycles)')
    ax2.set_title('RMSE -- a mean, comparable across datasets', fontsize=10)
    fig.suptitle(f'{SETTLED_CONFIG}, mean ensemble')
    return _save(fig, figure_dir(out_root), 'cross_dataset')


def plot_error_spread(per_subset, out_root):
    subsets = list(per_subset)
    data = [np.abs(per_subset[s]['pred'] - per_subset[s]['rul_true']) for s in subsets]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=subsets, patch_artist=True)
    for patch, colour in zip(bp['boxes'], [BLUE, ORANGE, GREEN, RED] * 4):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax.set_ylabel('|prediction error| (cycles)')
    ax.set_title(f'{SETTLED_CONFIG}: absolute error by dataset')
    return _save(fig, figure_dir(out_root), 'error_spread')


def _save(fig, directory, name):
    path = os.path.join(directory, f'{name}.png')
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def run(subsets, config=SETTLED_CONFIG, data_root=None, out_root=None):
    data_root = data_root or DATA_ROOT
    out_root = out_root or OUT_ROOT
    per_subset = {}

    for subset in subsets:
        preds, rul_true = load_predictions(subset, config, out_root, data_root)
        if preds is None:
            print(f"  {subset}: no ensemble predictions for {config} -- skipping. Run:\n"
                  f"    python -m ablation.evaluate --datasets {subset} --configs {config}")
            continue
        if 'mean' not in preds:
            print(f"  {subset}: predictions carry no 'mean' ensemble -- skipping")
            continue

        pred = preds['mean']
        per_subset[subset] = dict(pred=pred, rul_true=rul_true,
                                  metrics=all_metrics(pred, rul_true))
        written = [plot_error_hist(subset, pred, rul_true, out_root),
                   plot_scatter(subset, pred, rul_true, out_root),
                   plot_error_by_true(subset, pred, rul_true, out_root),
                   plot_ensembles(subset, preds, rul_true, out_root)]
        print(f"  {subset}: {len(written)} figures -> {os.path.dirname(written[0])}")

    if len(per_subset) >= 2:
        for path in (plot_cross_dataset(per_subset, out_root),
                     plot_error_spread(per_subset, out_root)):
            print(f"  cross-dataset: {path}")
    elif per_subset:
        print("  cross-dataset figures need >=2 datasets with predictions -- skipped")

    return per_subset


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--config', default=SETTLED_CONFIG)
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--out-root', default=OUT_ROOT)
    args = ap.parse_args()

    print("=" * 75)
    print(f"FIGURES -- {args.config}  datasets={args.datasets}")
    print("=" * 75)
    if not run(args.datasets, args.config, args.data_root, args.out_root):
        print("\nNothing drawn. Evaluate first:\n    python -m ablation.evaluate")


if __name__ == '__main__':
    main()
