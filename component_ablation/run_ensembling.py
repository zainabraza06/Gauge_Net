"""
COMPONENT-BY-COMPONENT ABLATION -- LOAD CHECKPOINTS + FULL ENSEMBLING SUITE
=============================================================================
NO TRAINING happens here (the training run disconnected partway through --
this script loads whatever checkpoints actually completed before that
happened, no hardcoded seed/config assumptions).

For each of the 8 cumulative configs (backbone_only -> +sensor_gate ->
near_only/far_only/near_far_both -> +reverse -> +spectral -> +healthy),
this script:
  1. Discovers whichever <config_name>_seed*.pt files actually exist on
     disk for each dataset (glob-based, not a hardcoded seed list --
     some configs may have fewer completed seeds than others).
  2. Reconstructs each seed's validation split deterministically from
     the seed (identical rng.choice call used during training), loads
     the checkpoint (built with the SAME config dict used to train it,
     so the architecture matches exactly), runs inference only.
  3. Applies the full ensembling suite (mean, median, trimmed_mean,
     val_weighted, ridge_stacking, softmax sharp/soft, rank_average
     [bug-fixed], bias_corrected) to whatever members were found for
     that config -- techniques requiring >=2 or >=3 members are
     skipped gracefully if fewer completed.

Architecture matches the confirmed-final settings: SENSOR_GATE_DIM=8,
SENSOR_GATE_HEADS=2, hard-window near-path (when active), absolute
position embedding, no patching, no fault-mode-discovery mechanism,
causal global+personal geometry, asym_huber + phase_weight loss only.
=============================================================================
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import glob
import json, pickle, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.linear_model import LinearRegression
from scipy.linalg import logm as _scipy_logm, sqrtm as _scipy_sqrtm

warnings.filterwarnings('ignore')

DATA_ROOT = '/content/drive/MyDrive/cmapps'
OUT_ROOT  = '/content/drive/MyDrive/cmapps/component_ablation'
os.makedirs(OUT_ROOT, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RUL_CAP = 125.0; SCALE = 125.0; W = 72; H = 96; DROPOUT = 0.15
LR = 3e-4; LR_MIN = 1e-6; WD = 1e-3; GRAD_CLIP = 0.5
EPOCHS = 150; WARMUP_EP = 20; BATCH = 128; VAL_FRAC = 0.20
END_TRIM_MAX = 5; PATIENCE = 25; BETA = 1.5; GAMMA = 1.5; HEALTHY_FRAC = 0.20
N_VAL_TRUNC = 8; MC_PASSES_VAL = 10; MC_PASSES_TEST = 100
USE_MULTIWINDOW = False; LAST_K = 9
SENSOR_GATE_DIM = 8; SENSOR_GATE_HEADS = 2
ATTN_HEADS = 2
NEAR_WINDOW = 8

SEEDS = [51, 52, 53, 54, 55]   # cosmetic only -- actual seeds used are auto-discovered per config via glob below
DATASETS = ['FD001', 'FD003']

CONFIGS = [
    dict(name='1_backbone_only', use_sensor_gate=False, path_mode='backbone',
         use_reverse=False, use_spectral=False, use_healthy=False),
    dict(name='2_plus_sensor_gate', use_sensor_gate=True, path_mode='backbone',
         use_reverse=False, use_spectral=False, use_healthy=False),
    dict(name='3a_near_only', use_sensor_gate=True, path_mode='near_only',
         use_reverse=False, use_spectral=False, use_healthy=False),
    dict(name='3b_far_only', use_sensor_gate=True, path_mode='far_only',
         use_reverse=False, use_spectral=False, use_healthy=False),
    dict(name='3c_near_far_both', use_sensor_gate=True, path_mode='both',
         use_reverse=False, use_spectral=False, use_healthy=False),
    dict(name='4_plus_reverse', use_sensor_gate=True, path_mode='both',
         use_reverse=True, use_spectral=False, use_healthy=False),
    dict(name='5_plus_spectral', use_sensor_gate=True, path_mode='both',
         use_reverse=True, use_spectral=True, use_healthy=False),
    dict(name='6_plus_healthy_FINAL', use_sensor_gate=True, path_mode='both',
         use_reverse=True, use_spectral=True, use_healthy=True),
]

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = '0'

print("=" * 75)
print("COMPONENT-BY-COMPONENT ABLATION -- cumulative additions")
print(f"Seeds: {SEEDS}  Configs: {[c['name'] for c in CONFIGS]}")
print("=" * 75)


def nasa_score(pred, target):
    e = np.array(pred) - np.array(target)
    return float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))

def rmse(pred, target):
    return float(np.sqrt(np.mean((np.array(pred) - np.array(target)) ** 2)))

def mae(pred, target):
    return float(np.mean(np.abs(np.array(pred) - np.array(target))))

def mean_error(pred, target):
    """Signed mean error -- positive means the model over-predicts RUL
    on average (the more dangerous direction), negative means it under-
    predicts. Worth reporting alongside MAE since MAE alone hides
    systematic bias direction."""
    return float(np.mean(np.array(pred) - np.array(target)))

def accuracy_within(pred, target, threshold):
    """Fraction of predictions within +/-threshold cycles of the true
    RUL -- the standard 'Accuracy(+/-K)' metric reported in RUL papers
    alongside RMSE/NASA score (e.g. Accuracy(+/-10), Accuracy(+/-20))."""
    e = np.abs(np.array(pred) - np.array(target))
    return float(np.mean(e <= threshold))

ACC_THRESHOLDS = [5, 10, 13, 15, 20]

def all_metrics(pred, target):
    """Computes every metric this run reports, for one prediction array."""
    m = dict(nasa=nasa_score(pred, target), rmse=rmse(pred, target),
              mae=mae(pred, target), mean_error=mean_error(pred, target))
    for k in ACC_THRESHOLDS:
        m[f'acc{k}'] = accuracy_within(pred, target, k)
    return m

def asym_huber(pred, target):
    e = pred - target; ae = e.abs()
    early = torch.where(ae <= 1., 0.5 * e ** 2, ae - 0.5)
    late = torch.where(ae <= 1., 0.5 * BETA * e ** 2, BETA * (ae - 0.5))
    return torch.where(e >= 0, late, early)

def phase_weight(phase, onset, cycle, T_i):
    psi = ((cycle - onset) / (T_i - onset).clamp(min=1.)).clamp(0., 1.)
    return 1. + GAMMA * psi * phase.float()


def _safe_corr_cov(X, eps=1e-6):
    Xc = X - X.mean(0)
    std = Xc.std(0); std = np.where(std < eps, eps, std)
    Xs = Xc / std
    return (Xs.T @ Xs) / max(len(X) - 1, 1) + eps * np.eye(X.shape[1])

def _causal_healthy_slice(g_sorted, cols, frac=HEALTHY_FRAC):
    nh = max(2, int(len(g_sorted) * frac))
    return g_sorted.iloc[:nh][cols].values.astype(np.float64)

def compute_ref_spd_single(tr_df, cols):
    all_h = []
    for uid, g in tr_df.groupby('unit', sort=False):
        all_h.append(_causal_healthy_slice(g.sort_values('cycle'), cols))
    return _safe_corr_cov(np.vstack(all_h))

def compute_was_ref_single(tr_df, cols):
    ref = {s: [] for s in cols}
    for uid, g in tr_df.groupby('unit', sort=False):
        h = _causal_healthy_slice(g.sort_values('cycle'), cols)
        for i, s in enumerate(cols):
            ref[s].extend(h[:, i].tolist())
    return {s: np.sort(np.array(v)) for s, v in ref.items()}

def precompute_riemannian(df, cols, ref_spd, w=W):
    ref_sqrt = _scipy_sqrtm(ref_spd).real
    try:
        ri = np.linalg.inv(ref_sqrt)
    except Exception:
        ri = np.eye(ref_spd.shape[0])
    vals = np.zeros(len(df))
    for uid, g in df.groupby('unit', sort=False):
        g_s = g.sort_values('cycle'); idx = g_s.index.values
        X = g_s[cols].values.astype(np.float64)
        for t in range(len(X)):
            win = X[max(0, t - w + 1):t + 1]
            if len(win) < 3:
                vals[idx[t]] = 0.; continue
            M = ri @ _safe_corr_cov(win) @ ri
            try:
                vals[idx[t]] = float(np.linalg.norm(_scipy_logm(M).real, 'fro'))
            except Exception:
                vals[idx[t]] = 0.
    return vals

def precompute_wasserstein_mean(df, cols, was_ref, w=W):
    vals = np.zeros(len(df))
    for uid, g in df.groupby('unit', sort=False):
        g_s = g.sort_values('cycle'); idx = g_s.index.values
        X = g_s[cols].values.astype(np.float64)
        for t in range(len(X)):
            win = X[max(0, t - w + 1):t + 1]; nw = len(win)
            per_sensor = np.zeros(len(cols))
            for si, sc in enumerate(cols):
                x = np.sort(win[:, si])
                q = np.quantile(was_ref[sc], np.linspace(0, 1, nw))
                per_sensor[si] = float(np.mean(np.abs(x - q)))
            vals[idx[t]] = per_sensor.mean()
    return vals

def precompute_riemannian_personal(df, cols, w=W):
    vals = np.zeros(len(df))
    for uid, g in df.groupby('unit', sort=False):
        g_s = g.sort_values('cycle'); idx = g_s.index.values
        X = g_s[cols].values.astype(np.float64); n = len(X)
        for t in range(n):
            n_ref = min(t + 1, w); reliability = min(n_ref / w, 1.0)
            ref_window = X[:n_ref]; current_window = X[max(0, t - w + 1):t + 1]
            if n_ref < 3 or len(current_window) < 3:
                vals[idx[t]] = 0.; continue
            try:
                ri = np.linalg.inv(_scipy_sqrtm(_safe_corr_cov(ref_window)).real)
                M = ri @ _safe_corr_cov(current_window) @ ri
                vals[idx[t]] = float(np.linalg.norm(_scipy_logm(M).real, 'fro')) * reliability
            except Exception:
                vals[idx[t]] = 0.
    return vals

def precompute_wasserstein_personal(df, cols, w=W):
    vals = np.zeros(len(df))
    for uid, g in df.groupby('unit', sort=False):
        g_s = g.sort_values('cycle'); idx = g_s.index.values
        X = g_s[cols].values.astype(np.float64); n = len(X)
        for t in range(n):
            n_ref = min(t + 1, w); reliability = min(n_ref / w, 1.0)
            ref_window = X[:n_ref]; current_window = X[max(0, t - w + 1):t + 1]
            nw_cur = len(current_window)
            if n_ref < 3 or nw_cur < 3:
                vals[idx[t]] = 0.; continue
            per_sensor = np.zeros(len(cols))
            for si in range(len(cols)):
                ref_sorted = np.sort(ref_window[:, si]); cur_sorted = np.sort(current_window[:, si])
                q = np.quantile(ref_sorted, np.linspace(0, 1, nw_cur))
                per_sensor[si] = float(np.mean(np.abs(cur_sorted - q)))
            vals[idx[t]] = per_sensor.mean() * reliability
    return vals

def build_geometry(traj_fn, tr_c, te_c, cache_dir):
    cache_path = os.path.join(cache_dir, 'geom_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache['geom_cols'], cache['tr_ext'], cache['te_ext']
    ref_spd = compute_ref_spd_single(tr_c, traj_fn)
    was_ref = compute_was_ref_single(tr_c, traj_fn)
    tr_ext = tr_c.copy(); te_ext = te_c.copy()
    tr_ext['riem_global'] = precompute_riemannian(tr_c, traj_fn, ref_spd)
    te_ext['riem_global'] = precompute_riemannian(te_c, traj_fn, ref_spd)
    tr_ext['was_global'] = precompute_wasserstein_mean(tr_c, traj_fn, was_ref)
    te_ext['was_global'] = precompute_wasserstein_mean(te_c, traj_fn, was_ref)
    tr_ext['riem_personal'] = precompute_riemannian_personal(tr_c, traj_fn)
    te_ext['riem_personal'] = precompute_riemannian_personal(te_c, traj_fn)
    tr_ext['was_personal'] = precompute_wasserstein_personal(tr_c, traj_fn)
    te_ext['was_personal'] = precompute_wasserstein_personal(te_c, traj_fn)
    geom_cols = ['riem_global', 'was_global', 'riem_personal', 'was_personal']
    with open(cache_path, 'wb') as f:
        pickle.dump(dict(geom_cols=geom_cols, tr_ext=tr_ext, te_ext=te_ext), f)
    return geom_cols, tr_ext, te_ext


class SensorRelationGate(nn.Module):
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

class SpectralGate(nn.Module):
    def __init__(self, n_f, h_dim=H):
        super().__init__(); n_low = max(1, W // 8)
        self.low_proj = nn.Linear(n_f, h_dim // 2); self.full_proj = nn.Linear(n_f, h_dim // 2)
        self.gate_out = nn.Linear(h_dim, h_dim); self.ln = nn.LayerNorm(h_dim); self.n_low = n_low
    def forward(self, x, h_time):
        spec = torch.fft.rfft(x.transpose(1, 2), dim=-1).abs()
        low_e = spec[:, :, 1:self.n_low + 1].mean(-1); full_e = spec[:, :, 1:].mean(-1)
        g = F.gelu(torch.cat([self.low_proj(low_e), self.full_proj(full_e)], dim=-1))
        return self.ln(h_time * torch.sigmoid(self.gate_out(g)).unsqueeze(1) + h_time)

class AttentionPooling(nn.Module):
    def __init__(self, h_dim):
        super().__init__(); self.score = nn.Linear(h_dim, 1)
    def forward(self, hm):
        a = self.score(hm); alpha = torch.softmax(a, dim=1)
        return torch.cat([(alpha * hm).sum(dim=1), hm[:, -1, :]], dim=-1)

class HealthyCrossAttentionPool(nn.Module):
    def __init__(self, h_dim, n_heads=2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(h_dim, n_heads, dropout=DROPOUT, batch_first=True)
        self.ln = nn.LayerNorm(h_dim)
    def forward(self, hm_current, hm_healthy, healthy_gate):
        ca, _ = self.cross_attn(query=hm_current, key=hm_healthy, value=hm_healthy)
        ca = self.ln(ca + hm_current)
        return torch.cat([ca.mean(dim=1), ca[:, -1, :]], dim=-1) * healthy_gate.unsqueeze(-1)


def _causal_mask(seq_len, window=None, device='cpu'):
    idx = torch.arange(seq_len, device=device)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(diff < 0, float('-inf'))
    if window is not None:
        mask.masked_fill_(diff > window, float('-inf'))
    return mask

class CausalSelfAttnFF(nn.Module):
    def __init__(self, h, n_heads=ATTN_HEADS, window=None):
        super().__init__()
        self.window = window
        self.attn = nn.MultiheadAttention(h, n_heads, dropout=DROPOUT, batch_first=True)
        self.ln1 = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, h * 2), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(h * 2, h))
        self.ln2 = nn.LayerNorm(h)
    def forward(self, x):
        seq_len = x.shape[1]
        mask = _causal_mask(seq_len, window=self.window, device=x.device)
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        return self.ln2(x + self.ff(x))

class TemporalBlock(nn.Module):
    """One temporal block class, parameterized by path_mode:
      'backbone'  : single full-causal path, no gating, no near path.
      'near_only' : single LOCAL-WINDOW causal path only.
      'far_only'  : single FULL-causal path only.
      'both'      : near + far, combined via the content-conditioned
                    gate (the dual-path design)."""
    def __init__(self, h, gate_dim, path_mode, near_window=NEAR_WINDOW, has_cross_attn=False):
        super().__init__()
        self.path_mode = path_mode
        if path_mode == 'near_only':
            self.near = CausalSelfAttnFF(h, window=near_window)
            self.far = None
        elif path_mode in ('far_only', 'backbone'):
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
            n = self.near(x); f = self.far(x)
            if self.has_ca:
                ca, _ = self.ca(query=n, key=f, value=f); n = self.ca_ln(n + ca)
            gate = torch.softmax(self.gate_proj(ctx), dim=-1)
            alpha = gate[:, 0:1].unsqueeze(1); beta = gate[:, 1:2].unsqueeze(1)
            return self.drop(alpha * n + beta * f) + self.res(x)
        elif self.path_mode == 'near_only':
            return self.drop(self.near(x)) + self.res(x)
        else:
            return self.drop(self.far(x)) + self.res(x)

class ReverseTransformerBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.block1 = CausalSelfAttnFF(h, window=None)
        self.block2 = CausalSelfAttnFF(h, window=None)
    def forward(self, x_flipped):
        h1 = self.block1(x_flipped)
        return self.block2(h1)


class IGRFNet_Ablation(nn.Module):
    def __init__(self, n_f, fml, h_dim=H, use_sensor_gate=True, path_mode='both',
                 use_reverse=True, use_spectral=True, use_healthy=True):
        super().__init__(); self.fml = fml; self.h_dim = h_dim
        self.use_sensor_gate = use_sensor_gate
        self.use_reverse = use_reverse
        self.use_spectral = use_spectral
        self.use_healthy = use_healthy
        gate_dim = 4

        self.sensor_gate = SensorRelationGate(n_f) if use_sensor_gate else None
        self.proj = nn.Sequential(nn.Linear(n_f, h_dim), nn.LayerNorm(h_dim), nn.GELU())

        self.blocks = nn.ModuleList([
            TemporalBlock(h_dim, gate_dim, path_mode, NEAR_WINDOW, False),
            TemporalBlock(h_dim, gate_dim, path_mode, NEAR_WINDOW, False),
            TemporalBlock(h_dim, gate_dim, path_mode, NEAR_WINDOW, True)])

        self.rev_backbone = ReverseTransformerBlock(h_dim) if use_reverse else None
        self.rev_blend = nn.Parameter(torch.tensor(-0.847)) if use_reverse else None

        self.sgate = SpectralGate(n_f, h_dim=h_dim) if use_spectral else None
        self.pos = nn.Parameter(torch.zeros(1, W, h_dim)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(h_dim, 2, dropout=DROPOUT, batch_first=True)
        self.aln = nn.LayerNorm(h_dim)
        self.attn_pool = AttentionPooling(h_dim)
        self.healthy_pool = HealthyCrossAttentionPool(h_dim) if use_healthy else None

        pooled_dim = h_dim * 4 if use_healthy else h_dim * 2
        self.head = nn.Sequential(nn.Linear(pooled_dim, h_dim // 2), nn.GELU(), nn.Dropout(DROPOUT),
                                   nn.Linear(h_dim // 2, h_dim // 4), nn.GELU(), nn.Linear(h_dim // 4, 1))
        nn.init.constant_(self.head[-1].bias, 0.64)

    def _gate_ctx(self, x, lf):
        ch_mean = x.mean(dim=-1)
        return torch.stack([ch_mean.mean(1), ch_mean.std(1) * 10., x.std(dim=-1).mean(1), lf], dim=-1)

    def _encode(self, x, lf):
        x_gated = self.sensor_gate(x) if self.use_sensor_gate else x
        ctx = self._gate_ctx(x_gated, lf)
        h1 = self.proj(x_gated)
        for b in self.blocks:
            h1 = b(h1, ctx)

        if self.use_reverse:
            h_rev = self.rev_backbone(self.proj(x_gated.flip(1)))
            alpha = torch.sigmoid(self.rev_blend)
            h1 = (1. - alpha) * h1 + alpha * h_rev

        hm = self.sgate(x_gated, h1) if self.use_spectral else h1
        pos = self.pos[:, :hm.shape[1], :]
        ha, _ = self.attn(hm + pos, hm + pos, hm + pos)
        return self.aln(ha + hm)

    def forward(self, x, lf, x_healthy=None, lf_healthy=None, healthy_gate=None):
        hm = self._encode(x, lf)
        feat_main = self.attn_pool(hm)
        if self.use_healthy and x_healthy is not None:
            hm_h = self._encode(x_healthy, lf_healthy if lf_healthy is not None else lf)
            hg = healthy_gate if healthy_gate is not None else torch.ones(hm.shape[0], device=hm.device)
            feat_healthy = self.healthy_pool(hm, hm_h, hg)
            feat = torch.cat([feat_main, feat_healthy], dim=-1)
        else:
            feat = feat_main
        return torch.clamp(self.head(feat).squeeze(-1), min=0.)


def _causal_current_window(feats, t, w=W):
    s = max(0, t - w + 1); win = feats[s:t + 1]; pad = w - len(win)
    if pad > 0: win = np.vstack([np.tile(feats[0], (pad, 1)), win])
    return win

def _causal_healthy_window(feats, t, w=W):
    n_avail = min(t + 1, w)
    healthy = feats[:n_avail]; pad = w - n_avail
    if pad > 0: healthy = np.vstack([healthy, np.tile(healthy[-1], (pad, 1))])
    return healthy.astype(np.float32), n_avail

def _healthy_gate_value(n_avail, w=W):
    return float(np.clip(n_avail / w, 0.0, 1.0))


class RULDataset(Dataset):
    def __init__(self, df, fn, fml, is_train, seed):
        self.fn = fn; self.fml = fml; self.rng = np.random.default_rng(seed)
        self.engines = []; self.index = []
        for uid, g in df.groupby('unit', sort=False):
            g = g.sort_values('cycle').reset_index(drop=True)
            if is_train and END_TRIM_MAX > 0:
                tr = int(self.rng.integers(0, END_TRIM_MAX + 1))
                if tr > 0 and len(g) > tr + W:
                    g = g.iloc[:-tr].reset_index(drop=True)
            n = len(g)
            rul = np.minimum(g['rul'].values.astype(np.float32) if 'rul' in g.columns else np.zeros(n), RUL_CAP)
            onset = (g['onset'].values.astype(np.float32) if 'onset' in g.columns else np.zeros(n))
            phase_loaded = (g['phase'].values.astype(np.float32) if 'phase' in g.columns else np.zeros(n, np.float32))
            if phase_loaded.sum() == 0:
                t_on = float(onset[0]) if len(onset) else 0.
                phase_loaded = (g['cycle'].values.astype(np.float32) >= t_on).astype(np.float32)
            self.engines.append({
                'feats': g[fn].values.astype(np.float32), 'rul': rul, 'phase': phase_loaded,
                'onset': onset, 'cycle': g['cycle'].values.astype(np.float32),
                'T_i': (g['T_i'].values.astype(np.float32) if 'T_i' in g.columns else np.full(n, float(n), np.float32))})
            ei = len(self.engines) - 1
            self.index.extend((ei, t) for t in range(n))
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        ei, t = self.index[idx]; e = self.engines[ei]
        win = _causal_current_window(e['feats'], t)
        hwin, n_avail = _causal_healthy_window(e['feats'], t)
        hgate = _healthy_gate_value(n_avail)
        lf = float(e['cycle'][t] / self.fml)
        hlf = float(e['cycle'][min(n_avail, len(e['cycle'])) - 1] / self.fml)
        return (torch.from_numpy(win), torch.tensor(e['rul'][t] / SCALE, dtype=torch.float32),
                {k: torch.tensor(e[k][t] / (SCALE if k in ('onset', 'T_i') else 1.), dtype=torch.float32)
                 for k in ['phase', 'onset', 'cycle', 'T_i']},
                torch.tensor(lf, dtype=torch.float32),
                torch.from_numpy(hwin), torch.tensor(hlf, dtype=torch.float32), torch.tensor(hgate, dtype=torch.float32))

def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            {k: torch.stack([b[2][k] for b in batch]) for k in batch[0][2]},
            torch.stack([b[3] for b in batch]),
            torch.stack([b[4] for b in batch]), torch.stack([b[5] for b in batch]), torch.stack([b[6] for b in batch]))


def build_val_snapshots(df_va, fn, fml, seed, n_trunc=N_VAL_TRUNC):
    vrng = np.random.default_rng(seed + 777)
    windows, ruls, lfs, hwins, hlfs, hgates = [], [], [], [], [], []
    for uid, g in df_va.groupby('unit', sort=False):
        g = g.sort_values('cycle').reset_index(drop=True)
        feats = g[fn].values.astype(np.float32)
        cyc = g['cycle'].values.astype(np.float32); cmax = float(cyc[-1])
        for _ in range(n_trunc):
            frac = vrng.uniform(0.20, 0.96); t_idx = int(np.argmin(np.abs(cyc - frac * cmax)))
            windows.append(_causal_current_window(feats, t_idx))
            hw, n_avail = _causal_healthy_window(feats, t_idx)
            hwins.append(hw); hgates.append(_healthy_gate_value(n_avail))
            ruls.append(min(cmax - cyc[t_idx], RUL_CAP)); lfs.append(cyc[t_idx] / fml)
            hlfs.append(cyc[min(n_avail, len(cyc)) - 1] / fml)
    return (torch.from_numpy(np.stack(windows).astype(np.float32)), torch.tensor(ruls, dtype=torch.float32),
            torch.tensor(lfs, dtype=torch.float32), torch.from_numpy(np.stack(hwins).astype(np.float32)),
            torch.tensor(hlfs, dtype=torch.float32), torch.tensor(hgates, dtype=torch.float32))

def mc_predict(model, x, lf, x_healthy=None, lf_healthy=None, healthy_gate=None, M=10):
    model.train()
    with torch.no_grad():
        preds = torch.stack([model(x, lf, x_healthy=x_healthy, lf_healthy=lf_healthy,
                                    healthy_gate=healthy_gate) for _ in range(M)], dim=0)
    model.eval()
    return preds.mean(0)

def predict_engine_lastK(model, feats, cycles, fml, device, M=MC_PASSES_TEST, k=LAST_K):
    n = len(feats); k_eff = k if USE_MULTIWINDOW else 1
    preds = []
    for end in range(max(1, n - k_eff + 1), n + 1):
        t = end - 1
        win = _causal_current_window(feats, t)
        hw, n_avail = _causal_healthy_window(feats, t)
        hg = _healthy_gate_value(n_avail)
        lf_val = float(cycles[t] / fml); hlf_val = float(cycles[min(n_avail, n) - 1] / fml)
        t_win = torch.from_numpy(win[np.newaxis].astype(np.float32)).to(device)
        hw_t = torch.from_numpy(hw[np.newaxis]).to(device)
        lf_t = torch.tensor([lf_val], dtype=torch.float32).to(device)
        hlf_t = torch.tensor([hlf_val], dtype=torch.float32).to(device)
        hg_t = torch.tensor([hg], dtype=torch.float32).to(device)
        p = float(mc_predict(model, t_win, lf_t, x_healthy=hw_t, lf_healthy=hlf_t, healthy_gate=hg_t, M=M).item()) * SCALE
        preds.append(p)
    preds = np.array(preds); ramp = np.linspace(1.0, len(preds), len(preds)); ramp /= ramp.sum()
    return float((preds * ramp).sum())


def load_and_evaluate_checkpoint(seed, fn, tr_c, te_c, rul_true, fml, ckpt, cfg, tag=''):
    """NO TRAINING -- loads an already-trained checkpoint (built with the
    exact same config dict used during training) and runs inference only.
    The validation split is reconstructed deterministically from the seed
    (identical rng.choice call used during training)."""
    rng = np.random.default_rng(seed)
    all_eng = sorted(tr_c['unit'].unique())
    n_val = max(1, int(len(all_eng) * VAL_FRAC))
    val_eng = set(int(u) for u in rng.choice(all_eng, n_val, replace=False))
    df_va = tr_c[tr_c['unit'].isin(val_eng)]

    model = IGRFNet_Ablation(len(fn), fml, h_dim=H,
                              use_sensor_gate=cfg['use_sensor_gate'], path_mode=cfg['path_mode'],
                              use_reverse=cfg['use_reverse'], use_spectral=cfg['use_spectral'],
                              use_healthy=cfg['use_healthy']).to(DEVICE)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Expected checkpoint at {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)

    val_win, val_rul, val_lf, val_hwin, val_hlf, val_hgate = build_val_snapshots(df_va, fn, fml, seed)
    val_win, val_rul, val_lf = val_win.to(DEVICE), val_rul.to(DEVICE), val_lf.to(DEVICE)
    val_hwin, val_hlf, val_hgate = val_hwin.to(DEVICE), val_hlf.to(DEVICE), val_hgate.to(DEVICE)

    with torch.no_grad():
        val_p = mc_predict(model, val_win, val_lf, x_healthy=val_hwin, lf_healthy=val_hlf,
                            healthy_gate=val_hgate, M=MC_PASSES_TEST).cpu().numpy() * SCALE
        best_vn = nasa_score(val_p.tolist(), val_rul.cpu().numpy().tolist())
    val_true_np = val_rul.cpu().numpy()

    # ── test-set inference, TIMED SEPARATELY from validation inference,
    # since inference time for a paper should reflect the actual
    # deployment-relevant cost (one engine's prediction), not validation
    # overhead which includes extra snapshot construction. ──
    n_test_engines = te_c['unit'].nunique()
    t_inf_start = time.time()
    preds = []
    for uid, g in te_c.groupby('unit', sort=True):
        gs = g.sort_values('cycle')
        f = gs[fn].values.astype(np.float32); cyc = gs['cycle'].values.astype(np.float32)
        preds.append(predict_engine_lastK(model, f, cyc, fml, DEVICE))
    inference_elapsed_s = time.time() - t_inf_start
    ms_per_engine = (inference_elapsed_s / max(n_test_engines, 1)) * 1000.0

    preds = np.array(preds)
    ns = nasa_score(preds, rul_true)
    rm = rmse(preds, rul_true)
    ma = mae(preds, rul_true)
    me = mean_error(preds, rul_true)
    acc_vals = {k: accuracy_within(preds, rul_true, k) for k in ACC_THRESHOLDS}

    acc_str = "  ".join(f"Acc(+/-{k})={acc_vals[k]*100:.1f}%" for k in ACC_THRESHOLDS)
    print(f"  [{tag}] params={npar:,}  seed={seed}  TEST_NASA={ns:.2f}  RMSE={rm:.3f}  MAE={ma:.3f}  "
          f"MeanErr={me:+.3f}  {acc_str}  "
          f"val_NASA={best_vn:.2f}  inference={ms_per_engine:.1f}ms/engine  ({n_test_engines} engines)")
    torch.cuda.empty_cache()
    result = dict(seed=seed, preds=preds, val_pred=val_p, val_true=val_true_np,
                  nasa=ns, rmse=rm, mae=ma, mean_error=me,
                  best_val_nasa=best_vn, n_params=npar,
                  ms_per_engine=ms_per_engine, n_test_engines=n_test_engines)
    for k, v in acc_vals.items():
        result[f'acc{k}'] = v
    return result


def discover_seeds(ckpt_dir, config_name):
    """Finds whichever <config_name>_seed*.pt files actually exist --
    no hardcoded seed list, since the run may have disconnected partway
    through and left some configs with fewer seeds than others."""
    pattern = os.path.join(ckpt_dir, f'{config_name}_seed*.pt')
    files = sorted(glob.glob(pattern))
    seeds = []
    for f in files:
        base = os.path.basename(f)
        seed_str = base.replace(f'{config_name}_seed', '').replace('.pt', '')
        try:
            seeds.append(int(seed_str))
        except ValueError:
            continue
    return seeds


def run_ensembling_suite(members, rul_true):
    """Returns {technique_name: full_metrics_dict_or_None} -- every
    technique's combined prediction array gets the FULL metric set
    (nasa, rmse, mae, mean_error, acc5/10/13/15/20), not just NASA/RMSE."""
    n_members = len(members)
    pred_matrix = np.stack([r['preds'] for r in members], axis=0)
    val_nasas = np.array([r['best_val_nasa'] for r in members])
    combined_preds = {}   # technique_name -> combined prediction array (or None if skipped)

    combined_preds['mean'] = pred_matrix.mean(axis=0)
    combined_preds['median'] = np.median(pred_matrix, axis=0)

    n_drop = max(1, int(n_members * 0.2)) if n_members >= 3 else 0
    if n_drop > 0:
        keep_idx = np.argsort(val_nasas)[:n_members - n_drop]
        combined_preds['trimmed_mean'] = pred_matrix[keep_idx].mean(axis=0)
    else:
        combined_preds['trimmed_mean'] = None

    inv_weights = 1.0 / np.maximum(val_nasas, 1e-6); inv_weights /= inv_weights.sum()
    combined_preds['val_weighted'] = (pred_matrix * inv_weights[:, None]).sum(axis=0)

    if n_members >= 2:
        val_pred_lens = [len(r['val_pred']) for r in members]
        if len(set(val_pred_lens)) == 1:
            from sklearn.linear_model import Ridge
            X_val = np.stack([r['val_pred'] for r in members], axis=1)
            y_val = members[0]['val_true']
            ridge = Ridge(alpha=1.0, positive=True).fit(X_val, y_val)
            combined_preds['ridge_stacking'] = np.clip(ridge.predict(pred_matrix.T), 0, None)
        else:
            combined_preds['ridge_stacking'] = None
    else:
        combined_preds['ridge_stacking'] = None

    for tau, name in [(50.0, 'softmax_sharp'), (150.0, 'softmax_soft')]:
        z = -val_nasas / tau
        sm_w = np.exp(z - z.max()); sm_w /= sm_w.sum()
        combined_preds[name] = (pred_matrix * sm_w[:, None]).sum(axis=0)

    # rank_average -- FIXED formula (interpolate within each engine's OWN
    # member predictions, never against a cross-engine pooled distribution)
    if n_members >= 2:
        ranks = np.argsort(np.argsort(pred_matrix, axis=0), axis=0).astype(np.float64)
        mean_rank = ranks.mean(axis=0)
        rank_preds = np.zeros(pred_matrix.shape[1])
        for i in range(pred_matrix.shape[1]):
            sorted_col = np.sort(pred_matrix[:, i])
            rank_preds[i] = np.interp(mean_rank[i], np.arange(n_members), sorted_col)
        combined_preds['rank_average'] = rank_preds
    else:
        combined_preds['rank_average'] = None

    mean_preds = combined_preds['mean']
    val_pred_pool = np.concatenate([r['val_pred'] for r in members])
    val_true_pool = np.concatenate([r['val_true'] for r in members])
    lr = LinearRegression().fit(val_pred_pool.reshape(-1, 1), val_true_pool - val_pred_pool)
    a, b = float(lr.coef_[0]), float(lr.intercept_)
    combined_preds['bias_corrected'] = np.clip(mean_preds + (a * mean_preds + b), 0, None)

    # ── compute the FULL metric set for every technique's combined predictions ──
    results = {}
    for name, preds in combined_preds.items():
        results[name] = all_metrics(preds, rul_true) if preds is not None else None
    return results


def run_dataset(SS):
    print(f"\n{'#'*75}\n# DATASET = {SS}\n{'#'*75}")
    out_dir = os.path.join(OUT_ROOT, SS)
    if not os.path.isdir(out_dir):
        print(f"  no checkpoint directory found at {out_dir} -- skipping")
        return None

    input_pkl = f'{DATA_ROOT}/pipeline_output/output_{SS}_v1.pkl' if SS == 'FD001' else \
                f'{DATA_ROOT}/pipeline_output/output_{SS}_v4.pkl'
    rul_file = f'{DATA_ROOT}/RUL_{SS}.txt'
    with open(input_pkl, 'rb') as f:
        out = pickle.load(f)
    traj_fn = out['trajectory_feature_names'][SS]
    base_fn = out['baseline_feature_names'][SS]
    tr_c_raw = out['feature_data'][SS]['train']; te_c_raw = out['feature_data'][SS]['test']
    geom_cols, tr_c, te_c = build_geometry(traj_fn, tr_c_raw, te_c_raw, out_dir)
    fn = traj_fn + base_fn + geom_cols
    print(f"  feature count: {len(fn)}")

    rul_true = np.minimum(pd.read_csv(rul_file, header=None, names=['rul']).values.ravel(), RUL_CAP)
    fml = float(np.median(tr_c.groupby('unit')['cycle'].max().values))

    all_ensembles = {}
    csv_rows = []   # flat rows for a paper-ready table: one row per (config, technique)
    for cfg in CONFIGS:
        name = cfg['name']
        seeds_found = discover_seeds(out_dir, name)
        print(f"\n{'='*70}\nCONFIG = {name}  (found seeds: {seeds_found})\n{'='*70}")
        if not seeds_found:
            print(f"  no checkpoints found for {name} -- skipping this config")
            continue

        members = []
        for seed in seeds_found:
            ckpt = os.path.join(out_dir, f'{name}_seed{seed}.pt')
            r = load_and_evaluate_checkpoint(seed, fn, tr_c, te_c, rul_true, fml, ckpt, cfg, tag=f'{SS}_{name}')
            members.append(r)

        indiv_nasa = np.array([r['nasa'] for r in members])
        indiv_ms = np.array([r['ms_per_engine'] for r in members])
        print(f"  [{name}] individual: NASA mean={indiv_nasa.mean():.2f} std={indiv_nasa.std():.2f}  "
              f"inference mean={indiv_ms.mean():.1f}ms/engine  (n={len(members)}, params={members[0]['n_params']:,})")

        ens = run_ensembling_suite(members, rul_true)
        all_ensembles[name] = dict(
            n_members=len(members), seeds_found=seeds_found, n_params=members[0]['n_params'],
            indiv_nasa_mean=float(indiv_nasa.mean()), indiv_nasa_std=float(indiv_nasa.std()),
            indiv_ms_per_engine_mean=float(indiv_ms.mean()),
            ensembles=ens)

        print(f"\n  [{name}] ensembling results (full metric set):")
        for technique, m in ens.items():
            if m is None:
                print(f"    {technique:<18}: SKIPPED")
                continue
            acc_str = "  ".join(f"Acc{k}={m[f'acc{k}']*100:.1f}%" for k in ACC_THRESHOLDS)
            print(f"    {technique:<18}: NASA={m['nasa']:>8.2f}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  "
                  f"MeanErr={m['mean_error']:+.3f}  {acc_str}")
            csv_rows.append(dict(dataset=SS, config=name, technique=technique, n_members=len(members),
                                  n_params=members[0]['n_params'],
                                  nasa=m['nasa'], rmse=m['rmse'], mae=m['mae'], mean_error=m['mean_error'],
                                  **{f'acc{k}': m[f'acc{k}'] for k in ACC_THRESHOLDS}))

    with open(os.path.join(out_dir, 'loaded_component_ensembles_full.json'), 'w') as f:
        json.dump(all_ensembles, f, indent=2)

    csv_path = os.path.join(out_dir, f'{SS}_full_metrics_table.csv')
    if csv_rows:
        cols = list(csv_rows[0].keys())
        with open(csv_path, 'w') as f:
            f.write(','.join(cols) + '\n')
            for row in csv_rows:
                f.write(','.join(str(row[c]) for c in cols) + '\n')
        print(f"\n  wrote paper-ready CSV table: {csv_path}")

    print(f"\n{'='*70}\n{SS} -- SUMMARY (mean-ensemble NASA per config)\n{'='*70}")
    for cfg in CONFIGS:
        name = cfg['name']
        if name in all_ensembles and all_ensembles[name]['ensembles']['mean'] is not None:
            v = all_ensembles[name]['ensembles']['mean']['nasa']
            print(f"  {name:<22}: n_members={all_ensembles[name]['n_members']}  mean_ensemble_NASA={v:.2f}")
        else:
            print(f"  {name:<22}: NOT FOUND")

    return dict(dataset=SS, configs=all_ensembles, csv_rows=csv_rows)


def run():
    all_summaries = {}
    all_csv_rows = []
    for SS in DATASETS:
        s = run_dataset(SS)
        if s is not None:
            all_summaries[SS] = s
            all_csv_rows.extend(s.get('csv_rows', []))

    combined_csv_path = os.path.join(OUT_ROOT, 'all_datasets_full_metrics_table.csv')
    if all_csv_rows:
        cols = list(all_csv_rows[0].keys())
        with open(combined_csv_path, 'w') as f:
            f.write(','.join(cols) + '\n')
            for row in all_csv_rows:
                f.write(','.join(str(row[c]) for c in cols) + '\n')
        print(f"\nwrote combined paper-ready CSV table (both datasets): {combined_csv_path}")

    print(f"\n{'='*75}\nFINAL SUMMARY -- all configs, both datasets (mean ensemble NASA)\n{'='*75}")
    for SS, s in all_summaries.items():
        print(f"\n  {SS}:")
        for cfg in CONFIGS:
            name = cfg['name']
            if name in s['configs'] and s['configs'][name]['ensembles'].get('mean') is not None:
                v = s['configs'][name]['ensembles']['mean']['nasa']
                print(f"    {name:<22}: {v:.2f}")
            else:
                print(f"    {name:<22}: not found")
    print("=" * 75)
    return all_summaries


if __name__ == '__main__':
    run()
