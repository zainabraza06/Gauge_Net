"""
Training objective: asymmetric Huber, weighted toward failure.

Both pieces encode the same prior as the NASA score -- predicting a longer
life than the engine actually has is the costly error -- but in a smooth,
differentiable form.
"""

import torch

from .config import BETA, WEIGHT_MAX, WEIGHT_GAMMA


def asym_huber(pred, target):
    """Huber loss whose LATE side (pred >= target, over-predicting RUL) is
    scaled by BETA. Returns the per-sample loss so it can be weighted before
    reduction."""
    e = pred - target
    ae = e.abs()
    early = torch.where(ae <= 1., 0.5 * e ** 2, ae - 0.5)
    late = torch.where(ae <= 1., 0.5 * BETA * e ** 2, BETA * (ae - 0.5))
    return torch.where(e >= 0, late, early)


def smooth_weight(rul_scaled, w_max=WEIGHT_MAX, gamma=WEIGHT_GAMMA):
    """Sample weight from the capped RUL alone -- the torch counterpart of
    preprocessing.pipeline.smooth_weight, and the single weighting scheme for
    every subset.

    `rul_scaled` is the target the network sees, RUL/SCALE, which is already
    the fraction of the cap remaining. So the fraction INTO degradation is
    simply 1 - rul_scaled: a row at or above the cap has rul_scaled == 1 and
    weight exactly 1.0, and weight rises smoothly to 1 + w_max as RUL -> 0.

    This replaced a phase weight computed from onset/T_i/phase columns. Those
    columns were emitted by the FD001 pipeline but not by FD003's, and the
    dataset silently substituted onset=0, phase=1 when they were absent -- so
    the two subsets were trained on different objectives from code that looked
    dataset-agnostic. Depending only on RUL removes the possibility.
    """
    frac = (1.0 - rul_scaled).clamp(0., 1.)
    return 1.0 + w_max * frac.pow(gamma)
