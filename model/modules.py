"""
Building blocks of IGRFNet.

Each block corresponds to one rung of the ablation ladder, so a component can
be switched off by not constructing it rather than by branching inside the
forward pass.

There is no spectral gate, no reverse-time backbone and no healthy-window
contrast here -- see model.igrfnet for why each left.

On the spectral gate specifically: An FFT band-energy gate on the temporal
features was ablated over 5 seeds on both subsets, in both its original and a
corrected placement, and moved NASA/RMSE/MAE no further than seed noise; the
one encouraging signal, lower seed variance on FD003, did not reproduce on
FD001. It is removed rather than kept switched off, since a component that
never earned its place is not a rung of the ladder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (W, DROPOUT, SENSOR_BRANCH_DIM, SENSOR_BRANCH_HEADS,
                     SENSOR_BRANCH_LAYERS, ATTN_HEADS, NEAR_WINDOW)


class SensorRelationshipBranch(nn.Module):
    """A separate branch that attends ACROSS sensors, iTransformer-style.

    The inversion is the point: a token is not a timestep but a SENSOR, formed
    by projecting that sensor's entire causal window at once. So every token
    already encodes a trend before any cross-sensor comparison happens, and
    the attention is asking which sensors are drifting together, not which
    readings co-occur at one instant.

    This replaced an inline gate that embedded one timestep's scalar readings
    and squashed the result to a multiplicative mask. That gate could only
    ever see a single instant, so it could not represent "these two sensors
    are drifting together" at all. Its output is left as tokens for
    CrossAttentionFusion rather than pooled here.
    """

    def __init__(self, n_f, window_len=W, d_model=SENSOR_BRANCH_DIM,
                 n_heads=SENSOR_BRANCH_HEADS, n_layers=SENSOR_BRANCH_LAYERS):
        super().__init__()
        self.token_embed = nn.Linear(window_len, d_model)
        self.sensor_id_embed = nn.Parameter(torch.zeros(1, n_f, d_model))
        nn.init.trunc_normal_(self.sensor_id_embed, std=0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                attn=nn.MultiheadAttention(d_model, n_heads, dropout=DROPOUT, batch_first=True),
                ln1=nn.LayerNorm(d_model),
                ff=nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Dropout(DROPOUT), nn.Linear(d_model * 2, d_model)),
                ln2=nn.LayerNorm(d_model),
            )) for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, x):
        """(batch, window, n_f) -> (batch, n_f, d_model)."""
        tokens = x.transpose(1, 2)                                  # the inversion
        h = self.token_embed(tokens) + self.sensor_id_embed
        for layer in self.layers:
            a, _ = layer['attn'](h, h, h)
            h = layer['ln1'](h + a)
            h = layer['ln2'](h + layer['ff'](h))
        return h


class CrossAttentionFusion(nn.Module):
    """Folds the sensor tokens into the pooled temporal feature.

    The temporal feature is the query and the sensor tokens are key/value, so
    the temporal representation asks which sensor relationships are relevant
    to it, and the fused vector keeps the temporal branch's width. Confirmed
    better than concatenating the two branches, which forces the head to
    weigh them with fixed parameters instead.
    """

    def __init__(self, h_dim, sensor_d_model=SENSOR_BRANCH_DIM, n_heads=2):
        super().__init__()
        self.kv_proj = nn.Linear(sensor_d_model, h_dim)
        self.cross_attn = nn.MultiheadAttention(h_dim, n_heads, dropout=DROPOUT, batch_first=True)
        self.ln = nn.LayerNorm(h_dim)

    def forward(self, main_feat, sensor_tokens):
        kv = self.kv_proj(sensor_tokens)
        fused, _ = self.cross_attn(main_feat.unsqueeze(1), kv, kv)
        return self.ln(fused.squeeze(1) + main_feat)


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


class LinearCausalAttnFF(nn.Module):
    """Causal LINEAR attention, O(n) in the window length.

    Softmax attention builds an n x n score matrix. Replacing the softmax with
    a positive feature map phi(q).phi(k)^T lets the sum be reassociated:
    running sums of phi(k) v^T and phi(k) carry the whole prefix, so each step
    reads the past through two cumulative tensors rather than by scoring every
    pair. Causality comes from the cumulative sum itself, not from a mask --
    position t reads prefix sums up to t and nothing later.

    elu(x)+1 is the feature map: positive everywhere, which the denominator
    needs, and cheap. The denominator is clamped because a query orthogonal to
    every accumulated key would otherwise divide by ~0.

    Used for the FAR path only. The near path spans 8 cycles, where an exact
    score matrix is small and cheap; the far path spans the whole window,
    which is where the quadratic term actually costs something.
    """

    def __init__(self, h, n_heads=ATTN_HEADS, eps=1e-6):
        super().__init__()
        if h % n_heads:
            raise ValueError(f"hidden width {h} must divide by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = h // n_heads
        self.eps = eps
        self.q_proj = nn.Linear(h, h)
        self.k_proj = nn.Linear(h, h)
        self.v_proj = nn.Linear(h, h)
        self.out_proj = nn.Linear(h, h)
        self.ln1 = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, h * 2), nn.GELU(), nn.Dropout(DROPOUT),
                                nn.Linear(h * 2, h))
        self.ln2 = nn.LayerNorm(h)
        self.drop = nn.Dropout(DROPOUT)

    @staticmethod
    def _feature_map(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        b, t, h = x.shape
        shape = (b, t, self.n_heads, self.d_head)
        q = self._feature_map(self.q_proj(x).view(shape).transpose(1, 2))
        k = self._feature_map(self.k_proj(x).view(shape).transpose(1, 2))
        v = self.v_proj(x).view(shape).transpose(1, 2)

        kv = torch.einsum('bhtd,bhte->bhtde', k, v).cumsum(dim=2)
        k_sum = k.cumsum(dim=2)
        num = torch.einsum('bhtd,bhtde->bhte', q, kv)
        den = torch.einsum('bhtd,bhtd->bht', q, k_sum).clamp_min(self.eps).unsqueeze(-1)
        a = self.out_proj((num / den).transpose(1, 2).reshape(b, t, h))
        x = self.ln1(x + self.drop(a))
        return self.ln2(x + self.ff(x))


class TemporalBlock(nn.Module):
    """Attention encoder block, parameterised by backbone mode:

      'far_only' : one full-causal attention path. The single-path reference.
      'both'     : a NEAR path (local window, recent dynamics) and a FAR path
                   (full history, long-range trend) mixed by a gate that is
                   conditioned on window content and life fraction, so the
                   network can lean on recent dynamics late in life and on
                   the long trend early on.

    With has_cross_attn, the near path additionally attends to the far path
    before mixing, letting local features be read in the context of the
    long-range trend. Only the last block uses it, and only under 'both':
    the cross-attention belongs to the dual-path design itself, so every rung
    built on 'both' carries it.

    far_attn_type picks how the far path reads the window -- 'full' for exact
    quadratic attention, 'linear' for the O(n) approximation. Only the far
    path is switchable: the near path spans 8 cycles, so its score matrix is
    small however it is computed.
    """

    def __init__(self, h, gate_dim, path_mode, near_window=NEAR_WINDOW, has_cross_attn=False,
                 far_attn_type='full'):
        super().__init__()
        self.path_mode = path_mode
        far = (LinearCausalAttnFF(h) if far_attn_type == 'linear'
               else CausalSelfAttnFF(h, window=None))
        if path_mode == 'far_only':
            self.near = None
            self.far = far
        else:  # 'both'
            self.near = CausalSelfAttnFF(h, window=near_window)
            self.far = far
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
