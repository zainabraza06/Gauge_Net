"""
IGRFNet -- the ablatable network.

Every optional component is a constructor flag, so a rung of the ablation
ladder is a model that never builds the parts it is not testing. That keeps
the parameter counts honest (a disabled component contributes none) and means
a checkpoint can only ever be reloaded into the architecture that produced it.
"""

import torch
import torch.nn as nn

from .config import H, W, DROPOUT, NEAR_WINDOW, DEVICE
from .modules import (SensorRelationshipBranch, CrossAttentionFusion, AttentionPooling,
                      HealthyCrossAttentionPool, TemporalBlock, ReverseTransformerBlock,
                      TCNBackbone)


class IGRFNet(nn.Module):
    def __init__(self, n_f, fml, h_dim=H, use_sensor_branch=True, backbone='both',
                 use_reverse=True, use_healthy=True):
        super().__init__()
        self.fml = fml
        self.h_dim = h_dim
        self.use_sensor_branch = use_sensor_branch
        self.backbone = backbone
        self.use_reverse = use_reverse
        self.use_healthy = use_healthy
        gate_dim = 4     # see _gate_ctx

        # The branch reads the raw window, not the projected one, and its
        # output is fused into the POOLED temporal feature -- which is why the
        # fusion is built at h_dim*2, the width AttentionPooling emits.
        self.sensor_branch = SensorRelationshipBranch(n_f) if use_sensor_branch else None
        self.cross_fusion = CrossAttentionFusion(h_dim * 2) if use_sensor_branch else None
        self.proj = nn.Sequential(nn.Linear(n_f, h_dim), nn.LayerNorm(h_dim), nn.GELU())

        # The bottom rung swaps the whole attention stack for a dilated
        # causal conv stack; the attention rungs keep three blocks, and only
        # the last cross-attends near to far, so the first two build the two
        # views and the last lets them interact before pooling.
        if backbone == 'tcn':
            self.tcn = TCNBackbone(h_dim)
            self.blocks = None
        else:
            self.tcn = None
            self.blocks = nn.ModuleList([
                TemporalBlock(h_dim, gate_dim, backbone, NEAR_WINDOW, False),
                TemporalBlock(h_dim, gate_dim, backbone, NEAR_WINDOW, False),
                TemporalBlock(h_dim, gate_dim, backbone, NEAR_WINDOW, True)])

        self.rev_backbone = ReverseTransformerBlock(h_dim) if use_reverse else None
        # sigmoid(-0.847) ~ 0.30: the reverse view starts as a minority
        # contribution and the blend is learned from there.
        self.rev_blend = nn.Parameter(torch.tensor(-0.847)) if use_reverse else None

        self.pos = nn.Parameter(torch.zeros(1, W, h_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(h_dim, 2, dropout=DROPOUT, batch_first=True)
        self.aln = nn.LayerNorm(h_dim)
        self.attn_pool = AttentionPooling(h_dim)
        self.healthy_pool = HealthyCrossAttentionPool(h_dim) if use_healthy else None

        # attn_pool emits 2*h; the healthy contrast adds another 2*h.
        pooled_dim = h_dim * 4 if use_healthy else h_dim * 2
        self.head = nn.Sequential(nn.Linear(pooled_dim, h_dim // 2), nn.GELU(), nn.Dropout(DROPOUT),
                                  nn.Linear(h_dim // 2, h_dim // 4), nn.GELU(),
                                  nn.Linear(h_dim // 4, 1))
        # 0.64 * SCALE ~ 80 cycles: start near the mean capped RUL so early
        # training is not spent walking the bias up from zero.
        nn.init.constant_(self.head[-1].bias, 0.64)

    def _gate_ctx(self, x, lf):
        """Four scalars summarising the window, used to condition the
        near/far gate: mean level, how much that level moves across the
        window, cross-sensor spread, and life fraction (cycle / median
        fleet life). The x10 on the level's std just puts it on a scale
        comparable to the others."""
        ch_mean = x.mean(dim=-1)
        return torch.stack([ch_mean.mean(1), ch_mean.std(1) * 10.,
                            x.std(dim=-1).mean(1), lf], dim=-1)

    def _encode(self, x, lf):
        ctx = self._gate_ctx(x, lf)
        h1 = self.proj(x)
        if self.backbone == 'tcn':
            h1 = self.tcn(h1)
        else:
            for b in self.blocks:
                h1 = b(h1, ctx)

        if self.use_reverse:
            h_rev = self.rev_backbone(self.proj(x.flip(1)))
            alpha = torch.sigmoid(self.rev_blend)
            h1 = (1. - alpha) * h1 + alpha * h_rev

        hm = h1
        pos = self.pos[:, :hm.shape[1], :]
        ha, _ = self.attn(hm + pos, hm + pos, hm + pos)
        return self.aln(ha + hm)

    def forward(self, x, lf, x_healthy=None, lf_healthy=None, healthy_gate=None):
        hm = self._encode(x, lf)
        feat_main = self.attn_pool(hm)
        if self.use_sensor_branch:
            feat_main = self.cross_fusion(feat_main, self.sensor_branch(x))
        if self.use_healthy and x_healthy is None:
            # The head was sized for the healthy contrast, so falling back to
            # the main features alone would fail deep inside nn.Linear with an
            # opaque shape error. Say what is actually missing.
            raise ValueError(
                "this model was built with use_healthy=True, so forward() requires "
                "x_healthy (and normally lf_healthy / healthy_gate); pass them, or "
                "build the model with use_healthy=False")
        if self.use_healthy:
            # The healthy window goes through the SAME encoder, so current and
            # healthy states are compared in one representation space.
            hm_h = self._encode(x_healthy, lf_healthy if lf_healthy is not None else lf)
            hg = healthy_gate if healthy_gate is not None else torch.ones(hm.shape[0], device=hm.device)
            feat = torch.cat([feat_main, self.healthy_pool(hm, hm_h, hg)], dim=-1)
        else:
            feat = feat_main
        # RUL is non-negative; the head predicts RUL/SCALE.
        return torch.clamp(self.head(feat).squeeze(-1), min=0.)


def build_model(n_features, fml, cfg, device=DEVICE):
    """Builds the architecture described by one ablation config dict.

    The single place that maps a config onto a model, so training, checkpoint
    loading and the stacking pass cannot drift into building different
    architectures for the same config name.
    """
    return IGRFNet(n_features, fml, h_dim=H,
                   use_sensor_branch=cfg['use_sensor_branch'], backbone=cfg['backbone'],
                   use_reverse=cfg['use_reverse'],
                   use_healthy=cfg['use_healthy']).to(device)
