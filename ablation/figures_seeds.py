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

"""

import numpy as np

BLUE, ORANGE, GREEN, RED = '#4C72B0', '#DD8452', '#55A868', '#C44E52'


def plot_seed_variance(plt, subset, data, save, metric='nasa'):
    scores = data.get('seed_scores')
    if not scores:
        return None
    vals = [m[metric] for m in scores]
    fig, ax = plt.subplots(figsize=(4.5, 5))
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(vals))
    ax.scatter(jitter, vals, color=BLUE, alpha=0.75, s=55, zorder=3, label='individual seeds')
    ax.hlines(np.mean(vals), -0.25, 0.25, color=BLUE, ls=':', lw=1)
    ax.scatter([0], [data['metrics'][metric]], color=ORANGE, marker='D', s=100, zorder=4,
               label='ensemble')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{subset}: seed spread vs the ensemble ({metric.upper()})')
    ax.legend(fontsize=8)
    return save(fig, f'seed_variance_{subset}_{metric}', subset)


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
