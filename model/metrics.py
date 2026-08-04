"""
Evaluation metrics for RUL prediction.

Pure NumPy and free of torch on purpose: the same functions score a single
seed, an ensemble, and anything pasted into a notebook later.
"""

import numpy as np

ACC_THRESHOLDS = [5, 10, 13, 15, 20]


def nasa_score(pred, target):
    """The PHM08 / C-MAPSS asymmetric score. Late predictions (pred > target,
    i.e. the engine fails sooner than predicted) are penalised on a e^(e/10)
    curve against e^(-e/13) for early ones, so the score is dominated by the
    dangerous direction. It is a SUM, so it also scales with the number of
    engines scored -- only compare it across equally sized test sets."""
    e = np.array(pred) - np.array(target)
    return float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))


def rmse(pred, target):
    return float(np.sqrt(np.mean((np.array(pred) - np.array(target)) ** 2)))


def mae(pred, target):
    return float(np.mean(np.abs(np.array(pred) - np.array(target))))


def mean_error(pred, target):
    """Signed mean error -- positive means the model over-predicts RUL on
    average (the more dangerous direction), negative means it under-predicts.
    Worth reporting alongside MAE, which hides the bias direction."""
    return float(np.mean(np.array(pred) - np.array(target)))


def mape(pred, target, eps=1.0):
    """Mean absolute percentage error, in percent.

    eps floors the denominator: true RUL reaches 0 at failure, and without a
    floor the engines closest to failure -- the ones the score cares about --
    would each contribute an unbounded term. Reported as a diagnostic only;
    model selection stays on validation NASA.
    """
    t = np.maximum(np.abs(np.array(target)), eps)
    return float(np.mean(np.abs(np.array(pred) - np.array(target)) / t) * 100)


def r2(pred, target):
    """Fraction of RUL variance explained. NaN when the targets are constant,
    since there is then no variance to explain rather than a perfect fit."""
    p = np.array(pred)
    t = np.array(target)
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot <= 0:
        return float('nan')
    return float(1 - np.sum((t - p) ** 2) / ss_tot)


def accuracy_within(pred, target, threshold):
    """Fraction of predictions within +/-threshold cycles of the true RUL --
    the Accuracy(+/-K) metric usually reported next to RMSE and NASA score."""
    e = np.abs(np.array(pred) - np.array(target))
    return float(np.mean(e <= threshold))


def all_metrics(pred, target):
    """Every metric this study reports, for one prediction array."""
    m = dict(nasa=nasa_score(pred, target), rmse=rmse(pred, target),
             mae=mae(pred, target), mean_error=mean_error(pred, target),
             mape=mape(pred, target), r2=r2(pred, target))
    for k in ACC_THRESHOLDS:
        m[f'acc{k}'] = accuracy_within(pred, target, k)
    return m
