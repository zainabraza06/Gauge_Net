"""
Training objective: asymmetric Huber, weighted by degradation phase.

Both pieces encode the same prior as the NASA score -- predicting a longer
life than the engine actually has is the costly error -- but in a smooth,
differentiable form.
"""

import torch

from .config import BETA, GAMMA


def asym_huber(pred, target):
    """Huber loss whose LATE side (pred >= target, over-predicting RUL) is
    scaled by BETA. Returns the per-sample loss so it can be weighted before
    reduction."""
    e = pred - target
    ae = e.abs()
    early = torch.where(ae <= 1., 0.5 * e ** 2, ae - 0.5)
    late = torch.where(ae <= 1., 0.5 * BETA * e ** 2, BETA * (ae - 0.5))
    return torch.where(e >= 0, late, early)


def phase_weight(phase, onset, cycle, T_i):
    """Ramps the sample weight from 1 up to 1 + GAMMA across the degradation
    phase. psi is how far the cycle has progressed from degradation onset to
    end of life; rows before onset (phase == 0) keep weight 1."""
    psi = ((cycle - onset) / (T_i - onset).clamp(min=1.)).clamp(0., 1.)
    return 1. + GAMMA * psi * phase.float()
