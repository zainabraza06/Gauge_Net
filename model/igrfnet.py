"""
IGRFNet -- the ablatable network.

Every optional component is a constructor flag, so a rung of the ablation
ladder is a model that never builds the parts it is not testing. That keeps
the parameter counts honest (a disabled component contributes none) and means
a checkpoint can only ever be reloaded into the architecture that produced it.
"""

import torch
import torch.nn as nn

from .config import H, W, DROPOUT, NEAR_WINDOW, N_REGIMES, DEVICE
from .modules import (SensorRelationshipBranch, CrossAttentionFusion, AttentionPooling,
                      HealthyCrossAttentionPool, TemporalBlock, ReverseTransformerBlock)


class IGRFNet(nn.Module):
    """With use_regime, the LAST input column is the regime id rather than a
    continuous feature: the multi-regime subsets are z-scored per regime,
    which removes each regime's mean and with it the information about which
    flight condition a row was recorded in. Embedding the id per timestep and
    adding it to the projected features hands that back, so the near/far gate
    and everything downstream can condition on the condition, not only on
    sensor values it has already been subtracted out of.

    Splitting by POSITION is what lets one feature tensor carry both, so the
    caller must put regime_id last -- model.data.features_for does.
    """

    def __init__(self, n_f, fml, h_dim=H, use_sensor_branch=True, backbone='both',
                 use_reverse=True, use_healthy=True, use_regime=False,
                 n_regimes=N_REGIMES):
        super().__init__()
        self.fml = fml
        self.h_dim = h_dim
        self.use_sensor_branch = use_sensor_branch
        self.backbone = backbone
        self.use_reverse = use_reverse
        self.use_healthy = use_healthy
        self.use_regime = use_regime
        self.n_regimes = n_regimes
        gate_dim = 4     # see _gate_ctx

        # Everything continuous is one column narrower when the regime id
        # rides along, so the projection and the sensor branch are built for
        # the continuous width, not the tensor width.
        n_cont = n_f - 1 if use_regime else n_f
        self.regime_embed = nn.Embedding(n_regimes, h_dim) if use_regime else None
        if use_regime:
            nn.init.trunc_normal_(self.regime_embed.weight, std=0.02)

        # The branch reads the raw window, not the projected one, and its
        # output is fused into the POOLED temporal feature -- which is why the
        # fusion is built at h_dim*2, the width AttentionPooling emits.
        self.sensor_branch = SensorRelationshipBranch(n_cont) if use_sensor_branch else None
        self.cross_fusion = CrossAttentionFusion(h_dim * 2) if use_sensor_branch else None
        self.proj = nn.Sequential(nn.Linear(n_cont, h_dim), nn.LayerNorm(h_dim), nn.GELU())

        # Three blocks; only the last cross-attends near to far, so the first
        # two build the two views and the last lets them interact before
        # pooling.
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

    def _split_regime(self, x):
        """Splits the trailing regime id off the feature tensor.

        The id is stored as a float alongside the features, so it is rounded
        and clamped rather than cast: a value outside [0, n_regimes-1] would
        otherwise index the embedding out of bounds, and clamping keeps a
        malformed id from taking down a whole run.
        """
        if not self.use_regime:
            return x, None
        return x[..., :-1], x[..., -1].round().long().clamp(0, self.n_regimes - 1)

    def _encode(self, x_cont, lf, regime_idx=None):
        ctx = self._gate_ctx(x_cont, lf)
        h1 = self.proj(x_cont)
        if regime_idx is not None:
            h1 = h1 + self.regime_embed(regime_idx)
        for b in self.blocks:
            h1 = b(h1, ctx)

        if self.use_reverse:
            h_rev_in = self.proj(x_cont.flip(1))
            if regime_idx is not None:
                # The reverse path reads the window backwards, so its regime
                # ids have to be reversed with it -- otherwise each timestep
                # would be told it was flown in some other timestep's regime.
                h_rev_in = h_rev_in + self.regime_embed(regime_idx.flip(1))
            h_rev = self.rev_backbone(h_rev_in)
            alpha = torch.sigmoid(self.rev_blend)
            h1 = (1. - alpha) * h1 + alpha * h_rev

        hm = h1
        pos = self.pos[:, :hm.shape[1], :]
        ha, _ = self.attn(hm + pos, hm + pos, hm + pos)
        return self.aln(ha + hm)

    def forward(self, x, lf, x_healthy=None, lf_healthy=None, healthy_gate=None):
        x_cont, regime_idx = self._split_regime(x)
        hm = self._encode(x_cont, lf, regime_idx)
        feat_main = self.attn_pool(hm)
        if self.use_sensor_branch:
            feat_main = self.cross_fusion(feat_main, self.sensor_branch(x_cont))
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
            xh_cont, h_regime = self._split_regime(x_healthy)
            hm_h = self._encode(xh_cont, lf_healthy if lf_healthy is not None else lf,
                                h_regime)
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
                   use_reverse=cfg['use_reverse'], use_healthy=cfg['use_healthy'],
                   use_regime=cfg.get('use_regime', False)).to(device)
