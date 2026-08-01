"""
Combining the seeds of one config into a single prediction.

Each seed is an ensemble MEMBER. The techniques differ in what they use to
combine members: nothing (mean, median), each member's validation score
(best_k_mean, val_weighted, softmax_*), only the ordering members induce
(rank_average), or a fitted combiner (ridge_stacking, bias_corrected).

A technique that cannot be fitted from the members found is skipped WITH A
REASON rather than dropped silently, so a missing row in the results table is
never ambiguous.

On validation-score weighting: each member's val NASA comes from ITS OWN
validation split, since every seed holds out different engines. The scores
are comparable enough to rank members, but they carry split difficulty as
well as model quality and should not be read as calibrated estimates.
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

from model.config import DEVICE, SCALE, MC_PASSES_TEST, STACK_SEED, MIN_STACK_ENGINES
from model.igrfnet import build_model
from model.inference import mc_predict
from model.metrics import all_metrics
from model.windows import build_val_snapshots


def build_stacking_features(members, fn, tr_c, fml, cfg, device=DEVICE):
    """Design matrix for ridge stacking.

    Every member is scored on ONE shared snapshot set drawn from the engines
    that appear in EVERY member's validation split. That gives two properties
    the fit needs:

      aligned    column i and the target vector describe the same engines at
                 the same truncation points. Members cannot simply contribute
                 their own val_pred: each seed draws a different validation
                 split AND different truncation points within it, so those
                 columns describe different engines. The row counts match
                 regardless -- int(n_engines * VAL_FRAC) * N_VAL_TRUNC is the
                 same for every seed -- so a shape check cannot detect it.
      leak-free  no member trained on those engines, so the blend weights are
                 not fitted on any member's training data.

    Returns (X, y, None) on success, or (None, None, reason) when the members
    hold out too few engines in common to fit on.
    """
    common = set(members[0]['val_units'])
    for m in members[1:]:
        common &= set(m['val_units'])
    if len(common) < MIN_STACK_ENGINES:
        return None, None, (f'only {len(common)} engine(s) held out by all '
                            f'{len(members)} members, need >={MIN_STACK_ENGINES}')

    df_st = tr_c[tr_c['unit'].isin(common)]
    win, rul, lf, hwin, hlf, hgate = build_val_snapshots(df_st, fn, fml, STACK_SEED)
    win, lf = win.to(device), lf.to(device)
    hwin, hlf, hgate = hwin.to(device), hlf.to(device), hgate.to(device)

    cols = []
    for m in members:
        model = build_model(len(fn), fml, cfg, device=device)
        model.load_state_dict(torch.load(m['ckpt'], map_location=device))
        model.eval()
        with torch.no_grad():
            cols.append(mc_predict(model, win, lf, x_healthy=hwin, lf_healthy=hlf,
                                   healthy_gate=hgate, M=MC_PASSES_TEST).cpu().numpy() * SCALE)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f"    stacking set: {len(common)} engines held out by all members "
          f"-> {len(cols[0])} aligned snapshots")
    return np.stack(cols, axis=1), rul.numpy(), None


def run_ensembling_suite(members, rul_true, stack_data=None):
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

    # ridge_stacking -- non-negative blend weights fitted on the shared,
    # leak-free snapshot set from build_stacking_features.
    if n_members < 2:
        combined['ridge_stacking'] = None
        skip_reasons['ridge_stacking'] = 'needs >=2 members'
    elif stack_data is None or stack_data[0] is None:
        combined['ridge_stacking'] = None
        skip_reasons['ridge_stacking'] = stack_data[2] if stack_data else 'no stacking set supplied'
    else:
        ridge = Ridge(alpha=1.0, positive=True).fit(stack_data[0], stack_data[1])
        combined['ridge_stacking'] = np.clip(ridge.predict(pred_matrix.T), 0, None)

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
