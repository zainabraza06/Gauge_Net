"""
IGRFNet -- the ablatable network.

Every optional component is a constructor flag, so a rung of the ablation
ladder is a model that never builds the parts it is not testing. That keeps
the parameter counts honest (a disabled component contributes none) and means
a checkpoint can only ever be reloaded into the architecture that produced it.

Two components that earlier ladders carried are gone from the settled model:

  the reverse-time encoding -- a second backbone reading the window backwards,
  blended in by a learned scalar. It doubled the temporal parameters and did
  not survive as the ladder was rebuilt around the sensor branch.

  the healthy-window contrast -- cross-attention between the current window
  and the engine's own earliest cycles. The sensor branch subsumes what it
  was for: both answer "how does this engine differ from healthy", and the
  branch does it from the window itself instead of needing a second encoded
  window, a padding gate, and three extra tensors on every batch.

  regime conditioning -- an embedding of the operating regime added per
  timestep on FD002/FD004. It is not in the staircase, so keeping the code
  would mean a path nothing builds and nothing measures.

All three are deleted rather than left as switched-off flags -- a flag no
config sets is a component the study has stopped measuring.
"""

import torch
import torch.nn as nn

from .config import H, W, DROPOUT, NEAR_WINDOW, DEVICE
from .modules import (SensorRelationshipBranch, CrossAttentionFusion, AttentionPooling,
                      TemporalBlock)


class IGRFNet(nn.Module):

    def __init__(self, n_f, fml, h_dim=H, use_sensor_branch=True, backbone='both',
                 far_attn_type='full'):
        super().__init__()
        self.fml = fml
        self.h_dim = h_dim
        self.use_sensor_branch = use_sensor_branch
        self.backbone = backbone
        gate_dim = 4     # see _gate_ctx

        # The branch reads the raw window, not the projected one, and its
        # output is fused into the POOLED temporal feature -- which is why the
        # fusion is built at h_dim*2, the width AttentionPooling emits.
        self.sensor_branch = SensorRelationshipBranch(n_f) if use_sensor_branch else None
        self.cross_fusion = CrossAttentionFusion(h_dim * 2) if use_sensor_branch else None
        self.proj = nn.Sequential(nn.Linear(n_f, h_dim), nn.LayerNorm(h_dim), nn.GELU())

        # Three blocks; only the last cross-attends near to far, so the first
        # two build the two views and the last lets them interact before
        # pooling.
        self.blocks = nn.ModuleList([
            TemporalBlock(h_dim, gate_dim, backbone, NEAR_WINDOW, i == 2,
                          far_attn_type=far_attn_type)
            for i in range(3)])

        self.pos = nn.Parameter(torch.zeros(1, W, h_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(h_dim, 2, dropout=DROPOUT, batch_first=True)
        self.aln = nn.LayerNorm(h_dim)
        self.attn_pool = AttentionPooling(h_dim)

        # attn_pool emits (weighted summary || last step), so 2*h.
        self.head = nn.Sequential(nn.Linear(h_dim * 2, h_dim // 2), nn.GELU(), nn.Dropout(DROPOUT),
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

    def forward(self, x, lf):
        ctx = self._gate_ctx(x, lf)

        h1 = self.proj(x)
        for b in self.blocks:
            h1 = b(h1, ctx)

        pos = self.pos[:, :h1.shape[1], :]
        ha, _ = self.attn(h1 + pos, h1 + pos, h1 + pos)
        hm = self.aln(ha + h1)

        feat = self.attn_pool(hm)
        if self.use_sensor_branch:
            feat = self.cross_fusion(feat, self.sensor_branch(x))
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
                   far_attn_type=cfg.get('far_attn_type', 'full')).to(device)
