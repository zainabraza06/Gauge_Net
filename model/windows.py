"""
Causal windowing, the training dataset, and validation snapshots.

Everything a sample sees is drawn from cycles <= t of its own engine. The
padding rules matter as much as the slicing: a window shorter than W is
left-padded with the engine's FIRST row (so the sequence still ends at the
current cycle, which is what the last-step pooling reads), while a healthy
window shorter than W is right-padded with its own last row (so it still
begins at cycle 1, which is what makes it a healthy reference).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import W, SCALE, RUL_CAP, END_TRIM_MAX, VAL_FRAC, N_VAL_TRUNC


def causal_current_window(feats, t, w=W):
    """The w cycles ending at t. Left-padded with the first row when the
    engine has not lived w cycles yet, so index -1 is always cycle t."""
    s = max(0, t - w + 1)
    win = feats[s:t + 1]
    pad = w - len(win)
    if pad > 0:
        win = np.vstack([np.tile(feats[0], (pad, 1)), win])
    return win


def causal_healthy_window(feats, t, w=W):
    """The engine's own earliest cycles, capped at both w and what has been
    observed by t. Right-padded with its last row, so the window still starts
    at cycle 1. Returns the padded window and how many real cycles it holds."""
    n_avail = min(t + 1, w)
    healthy = feats[:n_avail]
    pad = w - n_avail
    if pad > 0:
        healthy = np.vstack([healthy, np.tile(healthy[-1], (pad, 1))])
    return healthy.astype(np.float32), n_avail


def healthy_gate_value(n_avail, w=W):
    """How much of the healthy window is real rather than padding. Scales the
    healthy contrast so an engine seen for a few cycles contributes a damped
    signal instead of a confident comparison against padding."""
    return float(np.clip(n_avail / w, 0.0, 1.0))


def val_engines_for_seed(tr_c, seed, val_frac=VAL_FRAC):
    """The validation split a given seed trains with. Deterministic in the
    seed alone, so a split can be reconstructed later from a checkpoint's
    filename without storing it."""
    rng = np.random.default_rng(seed)
    all_eng = sorted(tr_c['unit'].unique())
    n_val = max(1, int(len(all_eng) * val_frac))
    return set(int(u) for u in rng.choice(all_eng, n_val, replace=False))


class RULDataset(Dataset):
    """One sample per (engine, cycle): the causal window ending at that cycle,
    its healthy counterpart, and the capped RUL target.

    During training each engine is randomly trimmed by up to END_TRIM_MAX
    cycles at the end. Train engines run to failure, so without this the model
    would only ever see final windows that sit exactly at RUL 0; the trim
    varies where the observable trajectory stops, which is the situation at
    test time.
    """

    def __init__(self, df, fn, fml, is_train, seed):
        self.fn = fn
        self.fml = fml
        self.rng = np.random.default_rng(seed)
        self.engines = []
        self.index = []
        for _, g in df.groupby('unit', sort=False):
            g = g.sort_values('cycle').reset_index(drop=True)
            if is_train and END_TRIM_MAX > 0:
                tr = int(self.rng.integers(0, END_TRIM_MAX + 1))
                if tr > 0 and len(g) > tr + W:
                    g = g.iloc[:-tr].reset_index(drop=True)
            n = len(g)
            rul = np.minimum(g['rul'].values.astype(np.float32) if 'rul' in g.columns
                             else np.zeros(n), RUL_CAP)
            onset = (g['onset'].values.astype(np.float32) if 'onset' in g.columns
                     else np.zeros(n))
            phase_loaded = (g['phase'].values.astype(np.float32) if 'phase' in g.columns
                            else np.zeros(n, np.float32))
            if phase_loaded.sum() == 0:
                t_on = float(onset[0]) if len(onset) else 0.
                phase_loaded = (g['cycle'].values.astype(np.float32) >= t_on).astype(np.float32)
            self.engines.append({
                'feats': g[fn].values.astype(np.float32), 'rul': rul, 'phase': phase_loaded,
                'onset': onset, 'cycle': g['cycle'].values.astype(np.float32),
                'T_i': (g['T_i'].values.astype(np.float32) if 'T_i' in g.columns
                        else np.full(n, float(n), np.float32))})
            ei = len(self.engines) - 1
            self.index.extend((ei, t) for t in range(n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ei, t = self.index[idx]
        e = self.engines[ei]
        win = causal_current_window(e['feats'], t)
        hwin, n_avail = causal_healthy_window(e['feats'], t)
        hgate = healthy_gate_value(n_avail)
        lf = float(e['cycle'][t] / self.fml)
        hlf = float(e['cycle'][min(n_avail, len(e['cycle'])) - 1] / self.fml)
        return (torch.from_numpy(win),
                torch.tensor(e['rul'][t] / SCALE, dtype=torch.float32),
                {k: torch.tensor(e[k][t] / (SCALE if k in ('onset', 'T_i') else 1.),
                                 dtype=torch.float32)
                 for k in ['phase', 'onset', 'cycle', 'T_i']},
                torch.tensor(lf, dtype=torch.float32),
                torch.from_numpy(hwin),
                torch.tensor(hlf, dtype=torch.float32),
                torch.tensor(hgate, dtype=torch.float32))


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            {k: torch.stack([b[2][k] for b in batch]) for k in batch[0][2]},
            torch.stack([b[3] for b in batch]),
            torch.stack([b[4] for b in batch]),
            torch.stack([b[5] for b in batch]),
            torch.stack([b[6] for b in batch]))


def build_val_snapshots(df_va, fn, fml, rng_seed, n_trunc=N_VAL_TRUNC):
    """Truncates each validation engine at n_trunc random points between 20%
    and 96% of its life, mimicking the test set, where engines are observed
    partway through. Validating on final windows only would score the model on
    a regime the test set never contains.

    Called with a member's own training seed to reproduce exactly the
    snapshots it early-stopped on, and with STACK_SEED to build the shared,
    member-independent snapshot set that ridge stacking is fitted on.
    """
    vrng = np.random.default_rng(rng_seed + 777)
    windows, ruls, lfs, hwins, hlfs, hgates = [], [], [], [], [], []
    for _, g in df_va.groupby('unit', sort=False):
        g = g.sort_values('cycle').reset_index(drop=True)
        feats = g[fn].values.astype(np.float32)
        cyc = g['cycle'].values.astype(np.float32)
        cmax = float(cyc[-1])
        for _ in range(n_trunc):
            frac = vrng.uniform(0.20, 0.96)
            t_idx = int(np.argmin(np.abs(cyc - frac * cmax)))
            windows.append(causal_current_window(feats, t_idx))
            hw, n_avail = causal_healthy_window(feats, t_idx)
            hwins.append(hw)
            hgates.append(healthy_gate_value(n_avail))
            ruls.append(min(cmax - cyc[t_idx], RUL_CAP))
            lfs.append(cyc[t_idx] / fml)
            hlfs.append(cyc[min(n_avail, len(cyc)) - 1] / fml)
    return (torch.from_numpy(np.stack(windows).astype(np.float32)),
            torch.tensor(ruls, dtype=torch.float32),
            torch.tensor(lfs, dtype=torch.float32),
            torch.from_numpy(np.stack(hwins).astype(np.float32)),
            torch.tensor(hlfs, dtype=torch.float32),
            torch.tensor(hgates, dtype=torch.float32))
