"""
Building blocks of IGRFNet.

Each block corresponds to one rung of the ablation ladder, so a component can
be switched off by not constructing it rather than by branching inside the
forward pass.

There is no spectral gate here. An FFT band-energy gate on the temporal
features was ablated over 5 seeds on both subsets, in both its original and a
corrected placement, and moved NASA/RMSE/MAE no further than seed noise; the
one encouraging signal, lower seed variance on FD003, did not reproduce on
FD001. It is removed rather than kept switched off, since a component that
never earned its place is not a rung of the ladder.
"""

import torch
import torch.nn as nn

from .config import DROPOUT, SENSOR_GATE_DIM, SENSOR_GATE_HEADS, ATTN_HEADS, NEAR_WINDOW


class SensorRelationGate(nn.Module):
    """Per-timestep attention ACROSS sensors, used to gate the inputs.

    Each sensor's scalar reading is embedded, given a learned channel
    identity, and attended against the other sensors at that same timestep;
    the result is squashed to a multiplicative gate in [0, 1]. So a sensor is
    amplified or suppressed according to what the other sensors read at that
    moment, which is what lets the network discount a channel that is noisy
    or uninformative in the current operating state. Attention is over the
    sensor axis only -- no information moves along time here.
    """

    def __init__(self, n_f, d_model=SENSOR_GATE_DIM, n_heads=SENSOR_GATE_HEADS):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.channel_embed = nn.Parameter(torch.zeros(1, n_f, d_model))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=DROPOUT, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        b, t, nf = x.shape
        xr = x.reshape(b * t, nf, 1)
        v = self.value_proj(xr) + self.channel_embed
        a, _ = self.attn(v, v, v)
        v2 = self.ln(v + a)
        return x * torch.sigmoid(self.out_proj(v2).reshape(b, t, nf))


class AttentionPooling(nn.Module):
    """Pools a sequence to (weighted summary || last step).

    Keeping the last step alongside the attention summary matters for RUL:
    the most recent cycle is the one the prediction is anchored to, and a
    pure attention average can dilute it.
    """

    def __init__(self, h_dim):
        super().__init__()
        self.score = nn.Linear(h_dim, 1)

    def forward(self, hm):
        a = self.score(hm)
        alpha = torch.softmax(a, dim=1)
        return torch.cat([(alpha * hm).sum(dim=1), hm[:, -1, :]], dim=-1)


class HealthyCrossAttentionPool(nn.Module):
    """Compares the current window against this engine's own healthy window.

    Cross-attending current (query) to healthy (key/value) makes the pooled
    feature a contrast rather than an absolute reading, which cancels the
    unit-to-unit build variation that makes absolute sensor levels a poor RUL
    cue. The healthy_gate scales the whole contribution by how much healthy
    history actually exists yet, so an engine seen for only a few cycles
    contributes a damped signal instead of a confident but ill-founded one.
    """

    def __init__(self, h_dim, n_heads=2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(h_dim, n_heads, dropout=DROPOUT, batch_first=True)
        self.ln = nn.LayerNorm(h_dim)

    def forward(self, hm_current, hm_healthy, healthy_gate):
        ca, _ = self.cross_attn(query=hm_current, key=hm_healthy, value=hm_healthy)
        ca = self.ln(ca + hm_current)
        return torch.cat([ca.mean(dim=1), ca[:, -1, :]], dim=-1) * healthy_gate.unsqueeze(-1)


def _causal_mask(seq_len, window=None, device='cpu'):
    """Additive attention mask. Always causal (no attending to the future);
    with window set, also blocks anything further back than `window` steps,
    which is what makes the near path local."""
    idx = torch.arange(seq_len, device=device)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(diff < 0, float('-inf'))
    if window is not None:
        mask.masked_fill_(diff > window, float('-inf'))
    return mask


class CausalSelfAttnFF(nn.Module):
    """Pre-norm-free transformer layer with a causal (optionally windowed)
    attention and a feed-forward, each residual and layer-normed."""

    def __init__(self, h, n_heads=ATTN_HEADS, window=None):
        super().__init__()
        self.window = window
        self.attn = nn.MultiheadAttention(h, n_heads, dropout=DROPOUT, batch_first=True)
        self.ln1 = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, h * 2), nn.GELU(), nn.Dropout(DROPOUT),
                                nn.Linear(h * 2, h))
        self.ln2 = nn.LayerNorm(h)

    def forward(self, x):
        mask = _causal_mask(x.shape[1], window=self.window, device=x.device)
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        return self.ln2(x + self.ff(x))


class TemporalBlock(nn.Module):
    """Temporal encoder block, parameterised by path_mode:

      'backbone' : one full-causal attention path. The ablation baseline.
      'both'     : a NEAR path (local window, recent dynamics) and a FAR path
                   (full history, long-range trend) mixed by a gate that is
                   conditioned on window content and life fraction, so the
                   network can lean on recent dynamics late in life and on
                   the long trend early on.

    With has_cross_attn, the near path additionally attends to the far path
    before mixing, letting local features be read in the context of the
    long-range trend. Only the last block uses it.
    """

    def __init__(self, h, gate_dim, path_mode, near_window=NEAR_WINDOW, has_cross_attn=False):
        super().__init__()
        self.path_mode = path_mode
        if path_mode == 'backbone':
            self.near = None
            self.far = CausalSelfAttnFF(h, window=None)
        else:  # 'both'
            self.near = CausalSelfAttnFF(h, window=near_window)
            self.far = CausalSelfAttnFF(h, window=None)
        self.gate_proj = nn.Linear(gate_dim, 2) if path_mode == 'both' else None
        self.res = nn.Linear(h, h)
        self.drop = nn.Dropout(DROPOUT)
        self.has_ca = has_cross_attn and path_mode == 'both'
        if self.has_ca:
            self.ca = nn.MultiheadAttention(h, 2, dropout=DROPOUT, batch_first=True)
            self.ca_ln = nn.LayerNorm(h)

    def forward(self, x, ctx):
        if self.path_mode == 'both':
            n = self.near(x)
            f = self.far(x)
            if self.has_ca:
                ca, _ = self.ca(query=n, key=f, value=f)
                n = self.ca_ln(n + ca)
            gate = torch.softmax(self.gate_proj(ctx), dim=-1)
            alpha = gate[:, 0:1].unsqueeze(1)
            beta = gate[:, 1:2].unsqueeze(1)
            return self.drop(alpha * n + beta * f) + self.res(x)
        return self.drop(self.far(x)) + self.res(x)


class ReverseTransformerBlock(nn.Module):
    """Encodes the window time-reversed.

    Read backwards, a degradation trajectory starts at its most-degraded end,
    so the reversed encoding emphasises how the engine ARRIVED at its current
    state rather than where it started. Its output is blended with the forward
    encoding by a learned scalar.
    """

    def __init__(self, h):
        super().__init__()
        self.block1 = CausalSelfAttnFF(h, window=None)
        self.block2 = CausalSelfAttnFF(h, window=None)

    def forward(self, x_flipped):
        return self.block2(self.block1(x_flipped))
