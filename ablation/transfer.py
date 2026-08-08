"""
Transfers a trained model from one subset to another. No architecture search.

    python -m ablation.transfer --source FD002 --target FD004
    python -m ablation.transfer --source FD002 --target FD004 --strategies frozen fine_tuned

Takes the best pretrained seed of the settled model on the SOURCE subset --
best meaning lowest validation NASA, read from the results the training stage
already wrote -- and adapts it to the TARGET subset under one or more
strategies, five target-side seeds each:

  frozen      everything except the head is frozen. Only the head trains, so
              the question it answers is whether the source representation is
              already good enough for a new subset to read off.
  fine_tuned  every parameter trains, at one learning rate.
  discrim_lr  every parameter trains, but the backbone at a tenth of the
              head's rate, on the usual argument that pretrained features
              need smaller steps than a head starting from scratch.

The head is REINITIALISED in every strategy, frozen included. It was fitted to
the source subset's RUL distribution, and a subset's engines differ in how
long they live, so carrying those weights over transfers a calibration rather
than a representation. Only the backbone is what transfer is being tested on.

Why the feature width lines up: every subset emits the same 14 sensors and the
same 4 geometry columns, so a source checkpoint loads into a target model with
no adapter. What differs between subsets is how the sensors were normalised
(single vs per-regime z-score), which is exactly the shift being transferred
across.

There is no from-scratch baseline here. The ladder already measures that:
train the same config on the target with `python -m ablation.train --datasets
<target> --configs sensor_branch_linear_far` and compare. Reporting transfer
numbers WITHOUT that comparison would say nothing about whether transfer
helped, so the summary prints the from-scratch result when it can find it.
"""

import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

from model.config import (DEVICE, DATA_ROOT, OUT_ROOT, SCALE, W, LR, LR_MIN, WD, GRAD_CLIP,
                          BATCH, MC_PASSES_VAL, MC_PASSES_TEST, patience_for)
from model.data import load_subset, features_for
from model.igrfnet import build_model
from model.inference import mc_predict, predict_test_set
from model.losses import asym_huber
from model.metrics import nasa_score, all_metrics, ACC_THRESHOLDS
from model.windows import RULDataset, collate, build_val_snapshots, val_engines_for_seed

from .configs import SEEDS, config_by_name
from .ensembling import run_ensembling_suite
from .stats import format_comparison, paired_comparison

# The config transferred. Not a parameter: the point of the exercise is the
# settled model, and a different architecture would need its own source run.
TRANSFER_CONFIG = 'sensor_branch_linear_far'

STRATEGIES = ('frozen', 'fine_tuned', 'discrim_lr')

# Fine-tuning is shorter than training from scratch -- the backbone starts
# useful, so the schedule only has to adapt it.
EPOCHS_FT = 80
WARMUP_FT = 10
LR_BACKBONE = LR / 10      # discriminative-LR strategy


def source_checkpoint(source, config_name, out_root=None):
    """The source seed with the lowest validation NASA, and its checkpoint.

    Reads the per-seed results the training stage wrote. Validation NASA is
    the right selector: it is the only score not measured on the test set, so
    picking on it does not leak the source's test engines into the transfer.
    """
    src_dir = os.path.join(out_root or OUT_ROOT, source)
    results_path = os.path.join(src_dir, f'{config_name}_results.pkl')
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"No source results at {results_path}\n\n"
            f"Train the source first:\n"
            f"    python -m ablation.train --datasets {source} --configs {config_name}")

    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    best = min(results, key=lambda r: r['best_val_nasa'])
    ckpt = os.path.join(src_dir, f"{config_name}_seed{best['seed']}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Source results name seed {best['seed']} but {ckpt} is missing")
    print(f"  source: {source} seed {best['seed']}  "
          f"(val NASA {best['best_val_nasa']:.2f}, the best of {len(results)}; "
          f"test NASA {best['nasa']:.2f})")
    return best['seed'], ckpt


def from_scratch_reference(target, config_name, out_root=None):
    """The same config trained on the target from scratch, if it exists.

    Transfer numbers on their own say nothing -- they need the alternative
    the practitioner actually has, which is training on the target directly.
    """
    path = os.path.join(out_root or OUT_ROOT, target, f'{config_name}_results.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def reinit_head(model):
    """Fresh head, the same way nn.Linear initialises itself, plus the bias
    the architecture starts from (0.64 * SCALE ~ 80 cycles)."""
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    nn.init.constant_(model.head[-1].bias, 0.64)


def split_parameters(model):
    """(backbone, head) parameter lists. Split by name so a component added
    to the model later lands in the backbone by default rather than being
    silently left out of both."""
    backbone = [p for n, p in model.named_parameters() if not n.startswith('head.')]
    head = [p for n, p in model.named_parameters() if n.startswith('head.')]
    return backbone, head


def build_optimiser(model, strategy):
    backbone, head = split_parameters(model)
    if strategy == 'frozen':
        for p in backbone:
            p.requires_grad = False
        return AdamW(head, lr=LR, weight_decay=WD)
    if strategy == 'discrim_lr':
        return AdamW([dict(params=backbone, lr=LR_BACKBONE), dict(params=head, lr=LR)],
                     weight_decay=WD)
    if strategy == 'fine_tuned':
        return AdamW(model.parameters(), lr=LR, weight_decay=WD)
    raise ValueError(f"unknown strategy {strategy!r}; known: {STRATEGIES}")


def transfer_one_seed(seed, fn, tr_c, te_c, rul_true, fml, ckpt_out, cfg, strategy,
                      pretrained, patience, tag=''):
    """Adapts one pretrained checkpoint to the target under one strategy."""
    val_eng = val_engines_for_seed(tr_c, seed)
    df_tr = tr_c[~tr_c['unit'].isin(val_eng)]
    df_va = tr_c[tr_c['unit'].isin(val_eng)]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = build_model(len(fn), fml, cfg)
    model.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    reinit_head(model)

    opt = build_optimiser(model, strategy)
    # Counted AFTER freezing, so the number describes what this run actually
    # trains rather than what the architecture contains.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    dl_tr = DataLoader(RULDataset(df_tr, fn, fml, True, seed), BATCH, shuffle=True,
                       collate_fn=collate, num_workers=0)
    val_win, val_rul, val_lf = build_val_snapshots(df_va, fn, fml, seed)
    val_win, val_rul, val_lf = val_win.to(DEVICE), val_rul.to(DEVICE), val_lf.to(DEVICE)

    sched = SequentialLR(opt, [LinearLR(opt, LR_MIN / LR, 1.0, WARMUP_FT),
                               CosineAnnealingLR(opt, T_max=EPOCHS_FT - WARMUP_FT, eta_min=LR_MIN)],
                         milestones=[WARMUP_FT])

    trainable = [p for p in model.parameters() if p.requires_grad]
    best_vn = float('inf')
    no_imp = 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, EPOCHS_FT + 1):
        model.train()
        for feats, rul_s, lf in dl_tr:
            feats, rul_s, lf = feats.to(DEVICE), rul_s.to(DEVICE), lf.to(DEVICE)
            loss = asym_huber(model(feats, lf), rul_s).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            p = mc_predict(model, val_win, val_lf, M=MC_PASSES_VAL)
            vn = nasa_score(p.cpu().numpy() * SCALE, val_rul.cpu().numpy())
        if vn < best_vn:
            best_vn, no_imp = vn, 0
            torch.save(model.state_dict(), ckpt_out)
        else:
            no_imp += 1
        if epoch % 20 == 0 or no_imp >= patience:
            print(f"    [{tag} seed{seed}] epoch {epoch:3d}/{EPOCHS_FT}  val_NASA={vn:.2f}  "
                  f"best={best_vn:.2f}  no_improve={no_imp}")
        if no_imp >= patience:
            break
    train_seconds = time.time() - t0

    model.load_state_dict(torch.load(ckpt_out, map_location=DEVICE))
    model.eval()
    preds, inference_seconds = predict_test_set(model, te_c, fn, fml)
    m = all_metrics(preds, rul_true)
    acc_str = "  ".join(f"Acc(+/-{k})={m[f'acc{k}']*100:.1f}%" for k in ACC_THRESHOLDS)
    print(f"  [{tag}] trainable={n_trainable:,}/{n_total:,}  seed={seed}  "
          f"TEST_NASA={m['nasa']:.2f}  RMSE={m['rmse']:.3f}  MAPE={m['mape']:.1f}%  "
          f"R2={m['r2']:.3f}  {acc_str}  best_val_NASA={best_vn:.2f}  epochs~{epoch}  "
          f"train={train_seconds / 60:.1f}min")

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return dict(seed=seed, best_val_nasa=best_vn, n_params=n_trainable, n_params_total=n_total,
                epochs=epoch, train_seconds=train_seconds,
                inference_seconds=inference_seconds, preds=preds, **m)


def efficiency_profile(model, n_features, n_repeats=200, warmup=20):
    """What one prediction costs: parameters, MACs, latency, peak memory.

    Reported for the architecture, not per seed -- seeds change the weights,
    and none of these four numbers depends on the weights. Latency is one
    forward pass; a real prediction runs MC_PASSES_TEST of them, so that
    product is what deployment actually pays.
    """
    model.eval()
    x = torch.randn(1, W, n_features, device=DEVICE)
    lf = torch.zeros(1, device=DEVICE)

    try:
        from thop import profile as thop_profile
        macs = float(thop_profile(model, inputs=(x, lf), verbose=False)[0])
        macs_source = 'thop'
    except ImportError:
        macs, macs_source = None, 'not measured (pip install thop)'

    with torch.no_grad():
        for _ in range(warmup):
            model(x, lf)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(n_repeats):
            model(x, lf)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    per_forward_ms = (elapsed / n_repeats) * 1000
    peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if DEVICE == 'cuda' else None
    return dict(
        total_params=sum(p.numel() for p in model.parameters()),
        macs=macs, macs_source=macs_source, flops=(None if macs is None else macs * 2),
        per_forward_ms=per_forward_ms,
        per_prediction_ms=per_forward_ms * MC_PASSES_TEST, mc_passes=MC_PASSES_TEST,
        peak_allocated_mb=peak_mb, device=DEVICE)


def run_pair(source, target, strategies, seeds, data_root=None, out_root=None,
             overwrite=False):
    print(f"\n{'#' * 75}\n# TRANSFER  {source} -> {target}\n{'#' * 75}")
    out_root = out_root or OUT_ROOT
    cfg = config_by_name(TRANSFER_CONFIG)

    _, source_ckpt = source_checkpoint(source, TRANSFER_CONFIG, out_root)

    target_dir = os.path.join(out_root, target)
    os.makedirs(target_dir, exist_ok=True)
    traj_fn, geom_cols, tr_c, te_c, rul_true, fml = load_subset(
        target, cache_dir=target_dir, data_root=data_root, out_root=out_root)
    fn = features_for(cfg, traj_fn, geom_cols)
    patience = patience_for(target)
    print(f"  target: {target}  ({tr_c['unit'].nunique()} train engines, "
          f"{te_c['unit'].nunique()} test engines, {len(fn)} features, patience {patience})")

    arms = {}
    for strategy in strategies:
        arm = f'{source}_to_{target}_{strategy}'
        print(f"\n{'=' * 70}\nARM = {arm}\n{'=' * 70}")
        arm_dir = os.path.join(out_root, 'transfer', arm)
        os.makedirs(arm_dir, exist_ok=True)

        members = []
        for seed in seeds:
            ckpt = os.path.join(arm_dir, f'seed{seed}.pt')
            if os.path.exists(ckpt) and not overwrite:
                print(f"  [{arm}] seed={seed}: checkpoint exists, skipping "
                      f"(use --overwrite to retrain)")
                continue
            members.append(transfer_one_seed(seed, fn, tr_c, te_c, rul_true, fml, ckpt, cfg,
                                             strategy, source_ckpt, patience, tag=arm))
            with open(os.path.join(arm_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(members, f)
        if members:
            arms[strategy] = members

    if not arms:
        print("\n  nothing trained this run")
        return {}

    report(source, target, arms, rul_true, fn, cfg, out_root)
    return arms


def report(source, target, arms, rul_true, fn, cfg, out_root):
    print(f"\n{'=' * 75}\n{source} -> {target}: per-strategy results\n{'=' * 75}")
    summary = {}
    for strategy, members in arms.items():
        nasa = np.array([r['nasa'] for r in members])
        ens, _, _ = run_ensembling_suite(members, rul_true)
        summary[strategy] = dict(
            n_seeds=len(members), trainable=members[0]['n_params'],
            total=members[0]['n_params_total'],
            nasa_mean=float(nasa.mean()), nasa_std=float(nasa.std()),
            rmse_mean=float(np.mean([r['rmse'] for r in members])),
            mean_ensemble=ens['mean'])
        s = summary[strategy]
        print(f"  {strategy:<12} trainable={s['trainable']:>9,}/{s['total']:,}  "
              f"NASA={s['nasa_mean']:>8.2f}+/-{s['nasa_std']:.2f}  "
              f"RMSE={s['rmse_mean']:.3f}  mean-ensemble NASA={s['mean_ensemble']['nasa']:.2f}")

    scratch = from_scratch_reference(target, TRANSFER_CONFIG, out_root)
    if scratch:
        # The comparison that decides whether transferring was worth it. Paired
        # by seed: both sides ran the same target-side seeds, so the pairing
        # removes the split-and-init variance rather than leaving it in.
        by_seed = {r['seed']: r for r in scratch}
        print(f"\n  vs the same config trained on {target} from scratch "
              f"(NASA {np.mean([r['nasa'] for r in scratch]):.2f} over {len(scratch)} seeds):")
        for strategy, members in arms.items():
            shared = sorted({r['seed'] for r in members} & set(by_seed))
            if len(shared) < 2:
                print(f"    {strategy:<12} SKIPPED (needs >=2 shared seeds, found {len(shared)})")
                continue
            m_by_seed = {r['seed']: r for r in members}
            cmp_ = paired_comparison([by_seed[s]['nasa'] for s in shared],
                                     [m_by_seed[s]['nasa'] for s in shared])
            print(f"    {strategy:<12} {format_comparison('nasa', cmp_)}")
    else:
        print(f"\n  no from-scratch run found for {target} -- these numbers say what "
              f"transfer scores,\n  not whether it beat training on {target} directly. Run:\n"
              f"    python -m ablation.train --datasets {target} --configs {TRANSFER_CONFIG}")

    best_strategy = min(summary, key=lambda s: summary[s]['nasa_mean'])
    print(f"\n  best strategy: {best_strategy} (NASA {summary[best_strategy]['nasa_mean']:.2f})")

    # Profiled from a fresh model of the same shape: the four numbers below
    # depend on the architecture and the feature width, not on which seed's
    # weights are loaded.
    prof = efficiency_profile(build_model(len(fn), 100.0, cfg), len(fn))
    print(f"\n  efficiency ({prof['device']}): {prof['total_params']:,} params  "
          f"{prof['per_forward_ms']:.2f} ms/forward  "
          f"-> {prof['per_prediction_ms']:.0f} ms per engine at {prof['mc_passes']} MC passes"
          + (f"  peak {prof['peak_allocated_mb']:.0f} MB" if prof['peak_allocated_mb'] else "")
          + (f"  MACs {prof['macs']:,.0f}" if prof['macs'] else f"  MACs {prof['macs_source']}"))

    out_path = os.path.join(out_root, 'transfer', f'{source}_to_{target}_summary.json')
    with open(out_path, 'w') as f:
        json.dump(dict(source=source, target=target, config=TRANSFER_CONFIG,
                       strategies=summary, best_strategy=best_strategy,
                       efficiency=prof), f, indent=2, default=float)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--source', default='FD002', help='subset to transfer FROM')
    ap.add_argument('--target', default='FD004', help='subset to transfer TO')
    ap.add_argument('--strategies', nargs='+', default=['frozen'], choices=STRATEGIES)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--out-root', default=OUT_ROOT)
    args = ap.parse_args()

    if args.source == args.target:
        ap.error("--source and --target must differ; transferring a subset to itself "
                 "measures nothing")

    print("=" * 75)
    print(f"TRANSFER LEARNING -- {TRANSFER_CONFIG}")
    print(f"device={DEVICE}  {args.source} -> {args.target}  strategies={args.strategies}")
    print(f"seeds={args.seeds}")
    print("=" * 75)
    run_pair(args.source, args.target, args.strategies, args.seeds,
             data_root=args.data_root, out_root=args.out_root, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
