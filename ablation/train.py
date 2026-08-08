"""
Trains every rung of the ablation ladder, for every seed, on every dataset.

    python -m ablation.train
    python -m ablation.train --datasets FD002 --configs sensor_branch_linear_far

Checkpoints go to OUT_ROOT/<dataset>/<config>_seed<seed>.pt, which is the
layout ablation.evaluate discovers. A rung that already has a checkpoint for a
seed is skipped unless --overwrite is given, so a run interrupted partway
through (Colab disconnects) resumes instead of retraining from scratch.

Model selection is on validation NASA rather than RMSE: the study reports NASA
as its headline metric, and the two disagree about which epoch is best because
NASA penalises late predictions far harder than early ones.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

from model.config import (DEVICE, DATA_ROOT, OUT_ROOT, SCALE, LR, LR_MIN, WD, GRAD_CLIP,
                          EPOCHS, WARMUP_EP, BATCH, PATIENCE, MC_PASSES_VAL, DATASETS,
                          patience_for)
from model.data import load_subset, features_for
from model.igrfnet import build_model
from model.inference import mc_predict, predict_test_set
from model.losses import asym_huber
from model.metrics import nasa_score, all_metrics, ACC_THRESHOLDS
from model.windows import RULDataset, collate, build_val_snapshots, val_engines_for_seed

from .configs import CONFIGS, SEEDS, config_by_name, configs_in_run_order


def train_one_seed(seed, fn, tr_c, te_c, rul_true, fml, ckpt, cfg, tag='', patience=PATIENCE):
    """Trains one (config, seed) member and returns its test metrics."""
    val_eng = val_engines_for_seed(tr_c, seed)
    df_tr = tr_c[~tr_c['unit'].isin(val_eng)]
    df_va = tr_c[tr_c['unit'].isin(val_eng)]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = build_model(len(fn), fml, cfg)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dl_tr = DataLoader(RULDataset(df_tr, fn, fml, True, seed), BATCH, shuffle=True,
                       collate_fn=collate, num_workers=0)
    val_win, val_rul, val_lf = build_val_snapshots(df_va, fn, fml, seed)
    val_win, val_rul, val_lf = val_win.to(DEVICE), val_rul.to(DEVICE), val_lf.to(DEVICE)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    # Warmup then cosine decay: the gate and blend parameters are unstable at
    # full LR from step 0, and the cosine tail lets the fit settle.
    sched = SequentialLR(opt, [LinearLR(opt, LR_MIN / LR, 1.0, WARMUP_EP),
                               CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_EP, eta_min=LR_MIN)],
                         milestones=[WARMUP_EP])

    best_vn = float('inf')
    no_imp = 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for feats, rul_s, lf in dl_tr:
            feats, rul_s, lf = feats.to(DEVICE), rul_s.to(DEVICE), lf.to(DEVICE)
            pred = model(feats, lf)
            # Unweighted: the loss ablation preferred this to both weighted
            # variants tried, so every row counts the same.
            loss = asym_huber(pred, rul_s).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            p = mc_predict(model, val_win, val_lf, M=MC_PASSES_VAL)
            vn = nasa_score(p.cpu().numpy() * SCALE, val_rul.cpu().numpy())
        if vn < best_vn:
            best_vn, no_imp = vn, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
        if epoch % 20 == 0 or no_imp >= patience:
            print(f"    [{tag} seed{seed}] epoch {epoch:3d}/{EPOCHS}  val_NASA={vn:.2f}  "
                  f"best={best_vn:.2f}  no_improve={no_imp}")
        if no_imp >= patience:
            break
    train_seconds = time.time() - t0

    # Score the SELECTED checkpoint, not the final epoch's weights. Inference
    # is timed separately from training: they answer different questions --
    # what the study cost to run, versus what predicting one engine costs in
    # deployment -- and a single wall-clock number conflates them.
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    preds, inference_seconds = predict_test_set(model, te_c, fn, fml)
    m = all_metrics(preds, rul_true)
    acc_str = "  ".join(f"Acc(+/-{k})={m[f'acc{k}']*100:.1f}%" for k in ACC_THRESHOLDS)
    print(f"  [{tag}] params={npar:,}  seed={seed}  TEST_NASA={m['nasa']:.2f}  "
          f"RMSE={m['rmse']:.3f}  MAPE={m['mape']:.1f}%  R2={m['r2']:.3f}  {acc_str}  "
          f"best_val_NASA={best_vn:.2f}  epochs~{epoch}  "
          f"train={train_seconds / 60:.1f}min  infer={inference_seconds / 60:.2f}min")

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    # preds ride along so the ensembling stage can combine raw predictions
    # rather than already-reduced metrics.
    return dict(seed=seed, best_val_nasa=best_vn, n_params=npar,
                epochs=epoch, train_seconds=train_seconds,
                inference_seconds=inference_seconds, preds=preds, **m)


def run_dataset(subset, configs, seeds, overwrite=False, data_root=None, out_root=None):
    print(f"\n{'#' * 75}\n# DATASET = {subset}\n{'#' * 75}")
    out_dir = os.path.join(out_root or OUT_ROOT, subset)
    os.makedirs(out_dir, exist_ok=True)

    traj_fn, geom_cols, tr_c, te_c, rul_true, fml = load_subset(
        subset, cache_dir=out_dir, data_root=data_root, out_root=out_root)
    patience = patience_for(subset)
    print(f"  train engines: {tr_c['unit'].nunique()}  "
          f"|  test engines: {te_c['unit'].nunique()}  |  patience: {patience} epochs")

    all_results = {}
    # Trained full-model-first; the analysis stages always walk CONFIGS order.
    for cfg in configs_in_run_order(configs):
        name = cfg['name']
        fn = features_for(cfg, traj_fn, geom_cols)
        print(f"\n{'=' * 70}\nCONFIG = {name}  ({len(fn)} features, "
              f"geometry={cfg['use_geometry']}, "
              f"far_attn={cfg.get('far_attn_type', 'full')})\n{'=' * 70}")
        all_results[name] = []
        for seed in seeds:
            ckpt = os.path.join(out_dir, f'{name}_seed{seed}.pt')
            if os.path.exists(ckpt) and not overwrite:
                print(f"  [{subset}_{name}] seed={seed}: checkpoint exists, skipping "
                      f"(use --overwrite to retrain)")
                continue
            r = train_one_seed(seed, fn, tr_c, te_c, rul_true, fml, ckpt, cfg,
                               tag=f'{subset}_{name}', patience=patience)
            all_results[name].append(r)
            with open(os.path.join(out_dir, f'{name}_results.pkl'), 'wb') as f:
                pickle.dump(all_results[name], f)
            # Raw per-engine predictions, one array per seed. Written after
            # every seed so an interrupted run still leaves the members it
            # finished in a form the ensembling stage can combine.
            np.savez(os.path.join(out_dir, f'{name}_preds.npz'),
                     **{f'seed{res["seed"]}': res['preds'] for res in all_results[name]})

    summary = {name: dict(
        nasa_mean=float(np.mean([r['nasa'] for r in rs])),
        nasa_std=float(np.std([r['nasa'] for r in rs])),
        val_nasa_mean=float(np.mean([r['best_val_nasa'] for r in rs])),
        train_seconds_mean=float(np.mean([r['train_seconds'] for r in rs])),
        train_seconds_total=float(np.sum([r['train_seconds'] for r in rs])),
        inference_seconds_mean=float(np.mean([r['inference_seconds'] for r in rs])),
        n_seeds=len(rs), n_params=rs[0]['n_params'])
        for name, rs in all_results.items() if rs}

    if summary:
        print(f"\n{'=' * 75}\n{subset} -- TRAINED THIS RUN\n{'=' * 75}")
        print(f"  {'Config':<24}{'TEST_NASA mean':>16}{'std':>10}{'seeds':>8}"
              f"{'params':>12}{'train min':>12}{'infer min':>12}")
        for name, s in summary.items():
            print(f"  {name:<24}{s['nasa_mean']:>16.2f}{s['nasa_std']:>10.2f}"
                  f"{s['n_seeds']:>8}{s['n_params']:>12,}"
                  f"{s['train_seconds_mean'] / 60:>12.1f}"
                  f"{s['inference_seconds_mean'] / 60:>12.2f}")
        with open(os.path.join(out_dir, f'{subset}_training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--configs', nargs='+', default=None,
                    help='config names to train (default: the full ladder)')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--overwrite', action='store_true',
                    help='retrain even where a checkpoint already exists')
    ap.add_argument('--data-root', default=DATA_ROOT,
                    help=f'raw data + pipeline_output (default: {DATA_ROOT})')
    ap.add_argument('--out-root', default=OUT_ROOT,
                    help=f'checkpoints and result tables (default: {OUT_ROOT})')
    args = ap.parse_args()

    configs = [config_by_name(n) for n in args.configs] if args.configs else CONFIGS
    print("=" * 75)
    print("COMPONENT-BY-COMPONENT ABLATION -- training")
    print(f"device={DEVICE}  datasets={args.datasets}  seeds={args.seeds}")
    print(f"configs={[c['name'] for c in configs]}")
    print(f"data_root={args.data_root}")
    print(f"out_root ={args.out_root}")
    if DEVICE == 'cpu':
        print("NOTE: no CUDA device found -- training will run on CPU, which is far "
              "slower.\n      Use --configs / --seeds to run a subset.")
    print("=" * 75)
    for subset in args.datasets:
        run_dataset(subset, configs, args.seeds, overwrite=args.overwrite,
                    data_root=args.data_root, out_root=args.out_root)


if __name__ == '__main__':
    main()
