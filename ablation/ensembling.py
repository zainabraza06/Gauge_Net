"""
Combining the seeds of one config into a single prediction.

Each seed is an ensemble MEMBER. The techniques differ in what they use to
combine members: nothing (mean, median), each member's validation score
(best_k_mean, val_weighted, softmax_*), only the ordering members induce
(rank_average), or a fitted recalibration (bias_corrected).

A technique that cannot be fitted from the members found is skipped WITH A
REASON rather than dropped silently, so a missing row in the results table is
never ambiguous.

On validation-score weighting: each member's val NASA comes from ITS OWN
validation split, since every seed holds out different engines. The scores
are comparable enough to rank members, but they carry split difficulty as
well as model quality and should not be read as calibrated estimates.

Ridge stacking is deliberately absent. Fitting per-member blend weights needs
a snapshot set that every member can be scored on and that none of them
trained on, i.e. the engines held out by ALL members. With five seeds holding
out 20% each that intersection is empty in practice, so the technique could
never be fitted on this design without either leaking a member's training
data into its own weight or misaligning the columns.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from model.metrics import all_metrics


def run_ensembling_suite(members, rul_true):
    """Returns (results, skip_reasons).

    results       {technique: full metric dict, or None if skipped}
    skip_reasons  {technique: why it was skipped}
    """
    n_members = len(members)
    n_items = len(rul_true)
    pred_matrix = np.stack([r['preds'] for r in members], axis=0)
    val_nasas = np.array([r['best_val_nasa'] for r in members])
    combined = {}
    skip_reasons = {}

    combined['mean'] = pred_matrix.mean(axis=0)
    combined['median'] = np.median(pred_matrix, axis=0)

    # best_k_mean -- mean of the members with the best validation NASA,
    # dropping the worst ~20%. Not a symmetric trimmed mean: nothing is
    # trimmed off the good end, so the name says which members survive.
    n_drop = max(1, int(n_members * 0.2)) if n_members >= 3 else 0
    if n_drop > 0:
        keep_idx = np.argsort(val_nasas)[:n_members - n_drop]
        combined['best_k_mean'] = pred_matrix[keep_idx].mean(axis=0)
    else:
        combined['best_k_mean'] = None
        skip_reasons['best_k_mean'] = 'needs >=3 members to drop one'

    # val_weighted -- weight inversely to validation NASA (lower is better).
    inv_w = 1.0 / np.maximum(val_nasas, 1e-6)
    combined['val_weighted'] = (pred_matrix * (inv_w / inv_w.sum())[:, None]).sum(axis=0)

    # softmax weighting over -val_NASA. tau sets how sharply the best member
    # dominates: 50 concentrates weight, 150 stays close to a plain mean.
    for tau, name in [(50.0, 'softmax_sharp'), (150.0, 'softmax_soft')]:
        z = -val_nasas / tau
        sm_w = np.exp(z - z.max())
        combined[name] = (pred_matrix * (sm_w / sm_w.sum())[:, None]).sum(axis=0)

    # rank_average -- each member ranks the ENGINES it scored, those ranks are
    # averaged across members, and the average rank is mapped back onto the
    # sorted mean-prediction scale. Using only each member's ORDERING means a
    # member whose RUL scale is miscalibrated but whose ordering is right
    # still contributes usefully.
    #
    # Ranking across MEMBERS instead (axis=0) is degenerate: every engine's
    # rank column is a permutation of 0..n_members-1, so the mean rank is the
    # constant (n_members-1)/2 for every engine and the result is identically
    # the median, whatever the predictions are.
    if n_members >= 2 and n_items >= 2:
        ranks = np.argsort(np.argsort(pred_matrix, axis=1), axis=1).astype(np.float64)
        combined['rank_average'] = np.interp(ranks.mean(axis=0), np.arange(n_items),
                                             np.sort(pred_matrix.mean(axis=0)))
    else:
        combined['rank_average'] = None
        skip_reasons['rank_average'] = 'needs >=2 members and >=2 engines'

    # bias_corrected -- one global affine recalibration of the mean ensemble,
    # fitted on how the validation error trends with the prediction. Pooling
    # members' validation sets IS valid here: each (val_pred, val_true) pair
    # stays internally aligned, and only the pooled trend is fitted, not a
    # per-member weight.
    mean_preds = combined['mean']
    val_pred_pool = np.concatenate([r['val_pred'] for r in members])
    val_true_pool = np.concatenate([r['val_true'] for r in members])
    lr = LinearRegression().fit(val_pred_pool.reshape(-1, 1), val_true_pool - val_pred_pool)
    a, b = float(lr.coef_[0]), float(lr.intercept_)
    combined['bias_corrected'] = np.clip(mean_preds + (a * mean_preds + b), 0, None)

    results = {name: (all_metrics(p, rul_true) if p is not None else None)
               for name, p in combined.items()}
    return results, skip_reasons
