"""
Figures about the seeds and the ensembling, rather than about one prediction.

Drawn by `python -m ablation.figures` alongside the per-dataset plots; kept in
their own module because they need the per-seed members, not just the combined
predictions.

  seed_variance_<metric>  every seed's score against the ensemble built from
                          them. "Ensembling beats the seeds" is usually
                          asserted from two averaged numbers; drawn this way
                          it either holds visibly or it does not, and the
                          spread of the cloud is the honest measure of how
                          much a single reported number means.

  seed_diversity_<SS>     correlation of the seeds' per-engine errors. This
                          is the MECHANISM behind ensembling rather than its
                          outcome: averaging helps to the extent that members
                          fail on different engines, so near-1 correlations
                          would explain a flat ensembling result far better
                          than any p-value.

  ensembling_forest       the reported technique against the plain mean, one
                          interval per dataset. The question is whether an
                          interval clears zero, which four intervals answer
                          faster than four rows of p-values.

  technique_ranks         average rank of each technique across datasets.
                          A critical-difference diagram belongs here, but the
                          Nemenyi distance at four datasets is wider than the
                          whole rank range -- it could only ever conclude "no
                          difference", which says more about having four
                          datasets than about the techniques. The observed
                          ranks are drawn instead, with that stated.
"""

import numpy as np
from scipy import stats as scipy_stats

from model.metrics import all_metrics

from .stats import paired_comparison

BLUE, ORANGE, GREEN, RED = '#4C72B0', '#DD8452', '#55A868', '#C44E52'


def plot_seed_variance(plt, per_subset, save, metric='nasa'):
    subsets = [s for s in per_subset if per_subset[s].get('seed_scores')]
    if not subsets:
        return None
    fig, ax = plt.subplots(figsize=(1.8 * len(subsets) + 3, 5))
    rng = np.random.default_rng(0)
    for i, subset in enumerate(subsets):
        vals = [m[metric] for m in per_subset[subset]['seed_scores']]
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.11, 0.11, len(vals)), vals,
                   color=BLUE, alpha=0.75, s=45, zorder=3,
                   label='individual seeds' if i == 0 else None)
        ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=BLUE, ls=':', lw=1)
        ax.scatter([i], [per_subset[subset]['metrics'][metric]], color=ORANGE, marker='D',
                   s=95, zorder=4, label='mean ensemble' if i == 0 else None)
    ax.set_xticks(range(len(subsets)), subsets)
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()}: seeds vs the ensemble built from them')
    ax.legend(fontsize=9)
    return save(fig, f'seed_variance_{metric}')


def plot_seed_diversity(plt, subset, seed_preds, rul_true, save):
    if len(seed_preds) < 2:
        return None
    seeds = sorted(seed_preds)
    errs = np.stack([seed_preds[s] - rul_true for s in seeds])
    corr = np.corrcoef(errs)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(seeds)), [f'seed {s}' for s in seeds], rotation=45, ha='right')
    ax.set_yticks(range(len(seeds)), [f'seed {s}' for s in seeds])
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(corr[i, j]) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, label='correlation of per-engine error')
    off = (corr.sum() - np.trace(corr)) / (corr.size - len(seeds))
    room = 'little room for averaging' if off > 0.9 else 'seeds fail on different engines'
    ax.set_title(f'{subset}: cross-seed error correlation\n'
                 f'mean off-diagonal r = {off:.2f}  ({room})')
    return save(fig, f'seed_diversity_{subset}', subset)


def plot_ensembling_forest(plt, per_subset, save, reported='val_weighted'):
    rows = []
    for subset, d in per_subset.items():
        preds = d.get('all_preds', {})
        if reported not in preds or 'mean' not in preds:
            continue
        # Sign: mean minus reported, so a positive interval means the
        # reported technique has the smaller error and wins.
        cmp_ = paired_comparison(np.abs(preds['mean'] - d['rul_true']),
                                 np.abs(preds[reported] - d['rul_true']))
        rows.append((subset, cmp_['mean_diff'], cmp_['bootstrap_ci95']))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.6 * len(rows)))
    for i, (_, diff, (lo, hi)) in enumerate(rows):
        colour = GREEN if lo > 0 else (RED if hi < 0 else '#888888')
        ax.plot([lo, hi], [i, i], color=colour, lw=2.5)
        ax.scatter([diff], [i], color=colour, s=65, zorder=3)
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel(f'|error| difference: mean - {reported}   (positive = {reported} wins)')
    ax.set_title(f'Does {reported} beat the plain mean?\n'
                 f'mean difference with 95% bootstrap CI, paired per engine')
    return save(fig, 'ensembling_forest')


def plot_technique_ranks(plt, per_subset, save):
    subsets = [s for s in per_subset if per_subset[s].get('all_preds')]
    if len(subsets) < 2:
        return None
    names = sorted(set.intersection(*(set(per_subset[s]['all_preds']) for s in subsets)))
    if len(names) < 2:
        return None
    ranks = np.asarray([
        scipy_stats.rankdata([all_metrics(per_subset[s]['all_preds'][n],
                                          per_subset[s]['rul_true'])['nasa'] for n in names])
        for s in subsets])
    order = np.argsort(ranks.mean(axis=0))
    fig, ax = plt.subplots(figsize=(8, 5))
    for pos, idx in enumerate(order):
        ax.scatter(ranks[:, idx], np.full(len(subsets), pos), color=BLUE, alpha=0.6, s=45)
        ax.scatter([ranks[:, idx].mean()], [pos], color=ORANGE, marker='D', s=90, zorder=3)
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    ax.set_xlabel(f'rank within a dataset (1 = best NASA), over {len(subsets)} datasets')
    ax.set_title('Ensembling techniques by average rank\n'
                 'orange = mean rank; a Nemenyi test needs more datasets than this')
    return save(fig, 'technique_ranks')
