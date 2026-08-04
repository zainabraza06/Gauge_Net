"""
Training objective: plain unweighted asymmetric Huber.

It encodes the same prior as the NASA score -- predicting a longer life than
the engine actually has is the costly error -- in a smooth, differentiable
form.

There is no sample weighting. Two schemes were tried on top of this loss and
neither beat it: smooth_weight, which ramped the weight up as an engine
approached failure, and asym_plus_nasa_surrogate, which added a differentiable
approximation of the NASA score itself. The asymmetry in BETA already carries
the late/early preference, and re-weighting the same rows by how close they
are to failure mostly re-states it.
"""

import torch

from .config import BETA


def asym_huber(pred, target):
    """Huber loss whose LATE side (pred >= target, over-predicting RUL) is
    scaled by BETA. Returns the per-sample loss, so the caller reduces it --
    training takes a plain .mean()."""
    e = pred - target
    ae = e.abs()
    early = torch.where(ae <= 1., 0.5 * e ** 2, ae - 0.5)
    late = torch.where(ae <= 1., 0.5 * BETA * e ** 2, BETA * (ae - 0.5))
    return torch.where(e >= 0, late, early)
