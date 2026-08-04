"""
Inference: MC-dropout prediction, per-engine test prediction, and scoring a
trained checkpoint.

No training happens here. A checkpoint is always rebuilt through
build_model(cfg), so an architecture mismatch fails loudly at load_state_dict
instead of quietly producing a differently shaped model.
"""

import glob
import os
import time

import numpy as np
import torch

from .config import (DEVICE, SCALE, MC_PASSES_TEST, LAST_K, USE_MULTIWINDOW)
from .igrfnet import build_model
from .metrics import nasa_score, ACC_THRESHOLDS, all_metrics
from .windows import (causal_current_window, causal_healthy_window, healthy_gate_value,
                      build_val_snapshots, val_engines_for_seed)


def mc_predict(model, x, lf, x_healthy=None, lf_healthy=None, healthy_gate=None, M=10):
    """Mean of M stochastic forward passes with dropout LEFT ON.

    model.train() is deliberate: it re-enables dropout so each pass samples a
    different sub-network, and averaging them both smooths the prediction and
    exposes its spread. Gradients stay off, and eval mode is restored on the
    way out so the caller's model state is unchanged.
    """
    model.train()
    with torch.no_grad():
        preds = torch.stack([model(x, lf, x_healthy=x_healthy, lf_healthy=lf_healthy,
                                   healthy_gate=healthy_gate) for _ in range(M)], dim=0)
    model.eval()
    return preds.mean(0)


def predict_engine(model, feats, cycles, fml, device=DEVICE, M=MC_PASSES_TEST, k=LAST_K):
    """One RUL value for one test engine, from the end of its trajectory.

    With USE_MULTIWINDOW the last k windows are predicted and combined under a
    linear ramp that weights the most recent window highest; otherwise only
    the final window is used (the default, and what the reported results use).
    """
    n = len(feats)
    k_eff = k if USE_MULTIWINDOW else 1
    preds = []
    for end in range(max(1, n - k_eff + 1), n + 1):
        t = end - 1
        win = causal_current_window(feats, t)
        hw, n_avail = causal_healthy_window(feats, t)
        hg = healthy_gate_value(n_avail)
        lf_val = float(cycles[t] / fml)
        hlf_val = float(cycles[min(n_avail, n) - 1] / fml)
        t_win = torch.from_numpy(win[np.newaxis].astype(np.float32)).to(device)
        hw_t = torch.from_numpy(hw[np.newaxis]).to(device)
        lf_t = torch.tensor([lf_val], dtype=torch.float32).to(device)
        hlf_t = torch.tensor([hlf_val], dtype=torch.float32).to(device)
        hg_t = torch.tensor([hg], dtype=torch.float32).to(device)
        p = float(mc_predict(model, t_win, lf_t, x_healthy=hw_t, lf_healthy=hlf_t,
                             healthy_gate=hg_t, M=M).item()) * SCALE
        preds.append(p)
    preds = np.array(preds)
    ramp = np.linspace(1.0, len(preds), len(preds))
    ramp /= ramp.sum()
    return float((preds * ramp).sum())


def predict_test_set(model, te_c, fn, fml, device=DEVICE):
    """Predicts every test engine, in sorted unit order so predictions line up
    with the RUL_*.txt ground truth. Returns (predictions, seconds elapsed)."""
    t0 = time.time()
    preds = []
    for _, g in te_c.groupby('unit', sort=True):
        gs = g.sort_values('cycle')
        f = gs[fn].values.astype(np.float32)
        cyc = gs['cycle'].values.astype(np.float32)
        preds.append(predict_engine(model, f, cyc, fml, device))
    return np.array(preds), time.time() - t0


def discover_seeds(ckpt_dir, config_name):
    """Whichever <config_name>_seed*.pt files actually exist.

    Glob-based rather than a fixed seed list: a Colab run can disconnect
    partway through and leave configs with unequal numbers of completed
    seeds, and the evaluation should report what was actually trained.
    """
    seeds = []
    for f in glob.glob(os.path.join(ckpt_dir, f'{config_name}_seed*.pt')):
        base = os.path.basename(f)
        seed_str = base.replace(f'{config_name}_seed', '').replace('.pt', '')
        try:
            seeds.append(int(seed_str))
        except ValueError:
            continue
    # Numeric sort, not the filename sort: 'seed51' precedes 'seed7'
    # lexically, which would report the seeds in a confusing order.
    return sorted(seeds)


def evaluate_checkpoint(seed, fn, tr_c, te_c, rul_true, fml, ckpt, cfg, tag='',
                        device=DEVICE, mc_passes=MC_PASSES_TEST):
    """Loads one trained checkpoint and scores it. No training.

    The validation split is reconstructed from the seed alone, using the same
    rng.choice the training run made, so a member's held-out engines are known
    after the fact -- which is what lets the ensembling stage find engines no
    member trained on.
    """
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Expected checkpoint at {ckpt}")

    val_eng = val_engines_for_seed(tr_c, seed)
    df_va = tr_c[tr_c['unit'].isin(val_eng)]

    model = build_model(len(fn), fml, cfg, device=device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)

    val_win, val_rul, val_lf, val_hwin, val_hlf, val_hgate = build_val_snapshots(df_va, fn, fml, seed)
    val_win, val_lf = val_win.to(device), val_lf.to(device)
    val_hwin, val_hlf, val_hgate = val_hwin.to(device), val_hlf.to(device), val_hgate.to(device)
    with torch.no_grad():
        val_p = mc_predict(model, val_win, val_lf, x_healthy=val_hwin, lf_healthy=val_hlf,
                           healthy_gate=val_hgate, M=mc_passes).cpu().numpy() * SCALE
    val_true = val_rul.numpy()
    val_nasa = nasa_score(val_p, val_true)

    # Test inference is timed on its own: the number worth reporting is the
    # deployment cost of predicting one engine, not validation overhead.
    n_test_engines = te_c['unit'].nunique()
    preds, elapsed_s = predict_test_set(model, te_c, fn, fml, device)
    ms_per_engine = (elapsed_s / max(n_test_engines, 1)) * 1000.0

    m = all_metrics(preds, rul_true)
    acc_str = "  ".join(f"Acc(+/-{k})={m[f'acc{k}']*100:.1f}%" for k in ACC_THRESHOLDS)
    print(f"  [{tag}] params={npar:,}  seed={seed}  TEST_NASA={m['nasa']:.2f}  "
          f"RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  MeanErr={m['mean_error']:+.3f}  "
          f"MAPE={m['mape']:.1f}%  R2={m['r2']:.3f}  {acc_str}  val_NASA={val_nasa:.2f}  "
          f"inference={ms_per_engine:.1f}ms/engine  ({n_test_engines} engines)")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return dict(seed=seed, preds=preds, val_pred=val_p, val_true=val_true,
                val_units=sorted(val_eng), ckpt=ckpt, n_params=npar,
                best_val_nasa=val_nasa, ms_per_engine=ms_per_engine,
                n_test_engines=n_test_engines, **m)
