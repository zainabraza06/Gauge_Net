"""
Combining the seeds of one config into a single prediction.

Each seed is an ensemble MEMBER. Five techniques, the set the study reports:

  mean, median          use nothing but the predictions themselves
  val_weighted          weight inversely to each member's validation NASA
  softmax_sharp/_soft   weight by softmax over -val_NASA, at two temperatures

A technique that cannot be fitted from the members found is skipped WITH A
REASON rather than dropped silently, so a missing row in the results table is
never ambiguous.

On validation-score weighting: each member's val NASA comes from ITS OWN
validation split, since every seed holds out different engines. The scores
are comparable enough to rank members, but they carry split difficulty as
well as model quality and should not be read as calibrated estimates.

Four candidates are NOT here, each for its own reason:

  trimmed_mean     dropping the worst members by validation score is a third
                   way of spending the same signal val_weighted and the
                   softmax pair already spend, and it never separated from
                   them; the reported suite keeps the two that differ in how
                   sharply they concentrate weight, not three that differ in
                   nothing.

  rank_average     combines members through their ORDERING of the engines
                   rather than their predicted values. A different kind of
                   answer to a different question, and outside the reported
                   suite.

  bias_corrected   fit an affine correction from pooled validation residuals
                   and applied it to test predictions -- a second model fitted
                   on data the members had already been selected on, a free
                   parameter no other technique gets, and one whose benefit
                   could not be separated from that advantage.

  ridge stacking   fitting per-member blend weights needs a snapshot set every
                   member can be scored on and none of them trained on, i.e.
                   the engines held out by ALL members. With five seeds
                   holding out 20% each that intersection is empty in
                   practice, so it could never be fitted on this design
                   without either leaking a member's training data into its
                   own weight or misaligning the columns.
"""

import numpy as np

from model.metrics import all_metrics


def run_ensembling_suite(members, rul_true):
    """Returns (results, skip_reasons).

    results       {technique: full metric dict, or None if skipped}
    skip_reasons  {technique: why it was skipped}
    combined      {technique: the combined prediction array, or None}

    The predictions come back alongside the metrics so a caller can test one
    technique against another on per-engine errors, which a reduced metric
    can no longer support.
    """
    pred_matrix = np.stack([r['preds'] for r in members], axis=0)
    val_nasas = np.array([r['best_val_nasa'] for r in members])
    combined = {}
    skip_reasons = {}

    combined['mean'] = pred_matrix.mean(axis=0)
    combined['median'] = np.median(pred_matrix, axis=0)

    # val_weighted -- weight inversely to validation NASA (lower is better).
    inv_w = 1.0 / np.maximum(val_nasas, 1e-6)
    combined['val_weighted'] = (pred_matrix * (inv_w / inv_w.sum())[:, None]).sum(axis=0)

    # softmax weighting over -val_NASA. tau sets how sharply the best member
    # dominates: 50 concentrates weight, 150 stays close to a plain mean.
    for tau, name in [(50.0, 'softmax_sharp'), (150.0, 'softmax_soft')]:
        z = -val_nasas / tau
        sm_w = np.exp(z - z.max())
        combined[name] = (pred_matrix * (sm_w / sm_w.sum())[:, None]).sum(axis=0)

    results = {name: (all_metrics(p, rul_true) if p is not None else None)
               for name, p in combined.items()}
    return results, skip_reasons, combined
