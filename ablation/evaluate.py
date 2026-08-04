"""
Scores trained checkpoints and applies the ensembling suite. No training.

    python -m ablation.evaluate
    python -m ablation.evaluate --datasets FD003

Discovers whichever <config>_seed*.pt files exist under OUT_ROOT/<dataset>,
so a training run that was interrupted is reported on exactly as far as it
actually got, rather than failing on an assumed seed list.

Writes, per dataset:
    loaded_component_ensembles_full.json   every metric, plus skip reasons
    <dataset>_full_metrics_table.csv       one row per (config, technique)
and a combined CSV across datasets at OUT_ROOT.
"""

import argparse
import json
import os

import numpy as np

from model.config import DEVICE, DATA_ROOT, OUT_ROOT, DATASETS
from model.data import load_subset, features_for
from model.inference import discover_seeds, evaluate_checkpoint
from model.metrics import ACC_THRESHOLDS

from .configs import CONFIGS, config_by_name
from .ensembling import run_ensembling_suite
from .stats import DIFF_CONVENTION, format_comparison, paired_comparison

CSV_COLS = (['dataset', 'config', 'technique', 'n_members', 'n_params',
             'nasa', 'rmse', 'mae', 'mean_error', 'mape', 'r2']
            + [f'acc{k}' for k in ACC_THRESHOLDS])


def write_csv(path, rows):
    if not rows:
        return False
    with open(path, 'w') as f:
        f.write(','.join(CSV_COLS) + '\n')
        for row in rows:
            f.write(','.join(str(row[c]) for c in CSV_COLS) + '\n')
    return True


DELTA_METRICS = [('nasa', 2), ('rmse', 3), ('mae', 3)]


def print_step_deltas(subset, configs, per_seed):
    """Each rung against the one below it, paired on shared seeds.

    Walks the configs in ladder order, whatever order they were trained in --
    a delta between non-adjacent rungs would fold several components together
    and could not be attributed to any of them.
    """
    print(f"\n{'=' * 70}\n{subset} -- STEP-BY-STEP DELTAS (each rung vs the one below, "
          f"paired by seed)\n{'=' * 70}")
    print(f"  sign: {DIFF_CONVENTION}")
    print("  n=5 seeds is low-powered: Wilcoxon cannot reach p<0.05 at n=5 at all, so read")
    print("  the effect size and whether the bootstrap interval excludes zero, not the p-values.")

    for prev_cfg, cur_cfg in zip(configs, configs[1:]):
        prev, cur = prev_cfg['name'], cur_cfg['name']
        if prev not in per_seed or cur not in per_seed:
            print(f"\n  {prev} -> {cur}: SKIPPED (no checkpoints for "
                  f"{prev if prev not in per_seed else cur})")
            continue
        shared = sorted(set(per_seed[prev]) & set(per_seed[cur]))
        if len(shared) < 2:
            print(f"\n  {prev} -> {cur}: SKIPPED (needs >=2 shared seeds, found "
                  f"{len(shared)})")
            continue
        d_params = (per_seed[cur][shared[0]]['n_params']
                    - per_seed[prev][shared[0]]['n_params'])
        print(f"\n  {prev} -> {cur}:  params {d_params:+,}  (seeds {shared})")
        for metric, precision in DELTA_METRICS:
            cmp_ = paired_comparison([per_seed[prev][s][metric] for s in shared],
                                     [per_seed[cur][s][metric] for s in shared])
            print(f"    {format_comparison(metric, cmp_, precision)}")


def run_dataset(subset, configs, data_root=None, out_root=None):
    print(f"\n{'#' * 75}\n# DATASET = {subset}\n{'#' * 75}")
    out_dir = os.path.join(out_root or OUT_ROOT, subset)
    if not os.path.isdir(out_dir):
        print(f"  no checkpoint directory at {out_dir} -- skipping "
              f"(run 'python -m ablation.train' first)")
        return None

    traj_fn, geom_cols, tr_c, te_c, rul_true, fml = load_subset(
        subset, cache_dir=out_dir, data_root=data_root, out_root=out_root)

    all_ensembles = {}
    csv_rows = []
    per_seed = {}
    for cfg in configs:
        name = cfg['name']
        fn = features_for(cfg, traj_fn, geom_cols)
        seeds_found = discover_seeds(out_dir, name)
        print(f"\n{'=' * 70}\nCONFIG = {name}  ({len(fn)} features, "
              f"found seeds: {seeds_found})\n{'=' * 70}")
        if not seeds_found:
            print(f"  no checkpoints found for {name} -- skipping this config")
            continue

        members = [evaluate_checkpoint(seed, fn, tr_c, te_c, rul_true, fml,
                                       os.path.join(out_dir, f'{name}_seed{seed}.pt'),
                                       cfg, tag=f'{subset}_{name}')
                   for seed in seeds_found]

        # Keyed by seed, not positional: two rungs can have different seeds
        # present after an interrupted run, and the delta table must pair
        # like with like rather than by list position.
        per_seed[name] = {r['seed']: r for r in members}

        indiv_nasa = np.array([r['nasa'] for r in members])
        indiv_ms = np.array([r['ms_per_engine'] for r in members])
        print(f"  [{name}] individual: NASA mean={indiv_nasa.mean():.2f} "
              f"std={indiv_nasa.std():.2f}  inference mean={indiv_ms.mean():.1f}ms/engine  "
              f"(n={len(members)}, params={members[0]['n_params']:,})")

        ens, skips = run_ensembling_suite(members, rul_true)

        all_ensembles[name] = dict(
            n_members=len(members), seeds_found=seeds_found,
            n_params=members[0]['n_params'],
            indiv_nasa_mean=float(indiv_nasa.mean()), indiv_nasa_std=float(indiv_nasa.std()),
            indiv_ms_per_engine_mean=float(indiv_ms.mean()),
            ensembles=ens, skip_reasons=skips)

        print(f"\n  [{name}] ensembling results:")
        for technique, m in ens.items():
            if m is None:
                print(f"    {technique:<16}: SKIPPED ({skips.get(technique, 'unknown reason')})")
                continue
            acc_str = "  ".join(f"Acc{k}={m[f'acc{k}'] * 100:.1f}%" for k in ACC_THRESHOLDS)
            print(f"    {technique:<16}: NASA={m['nasa']:>8.2f}  RMSE={m['rmse']:.3f}  "
                  f"MAE={m['mae']:.3f}  MeanErr={m['mean_error']:+.3f}  "
                  f"MAPE={m['mape']:.1f}%  R2={m['r2']:.3f}  {acc_str}")
            csv_rows.append(dict(dataset=subset, config=name, technique=technique,
                                 n_members=len(members), n_params=members[0]['n_params'],
                                 **{k: m[k] for k in
                                    ['nasa', 'rmse', 'mae', 'mean_error', 'mape', 'r2']},
                                 **{f'acc{k}': m[f'acc{k}'] for k in ACC_THRESHOLDS}))

    print_step_deltas(subset, configs, per_seed)

    with open(os.path.join(out_dir, 'loaded_component_ensembles_full.json'), 'w') as f:
        json.dump(all_ensembles, f, indent=2)
    csv_path = os.path.join(out_dir, f'{subset}_full_metrics_table.csv')
    if write_csv(csv_path, csv_rows):
        print(f"\n  wrote metrics table: {csv_path}")

    print(f"\n{'=' * 70}\n{subset} -- mean-ensemble NASA per config\n{'=' * 70}")
    for cfg in configs:
        name = cfg['name']
        entry = all_ensembles.get(name)
        if entry and entry['ensembles'].get('mean'):
            print(f"  {name:<22}: n_members={entry['n_members']}  "
                  f"mean_ensemble_NASA={entry['ensembles']['mean']['nasa']:.2f}")
        else:
            print(f"  {name:<22}: NOT FOUND")

    return dict(dataset=subset, configs=all_ensembles, csv_rows=csv_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--configs', nargs='+', default=None)
    ap.add_argument('--data-root', default=DATA_ROOT,
                    help=f'raw data + pipeline_output (default: {DATA_ROOT})')
    ap.add_argument('--out-root', default=OUT_ROOT,
                    help=f'checkpoints and result tables (default: {OUT_ROOT})')
    args = ap.parse_args()
    configs = [config_by_name(n) for n in args.configs] if args.configs else CONFIGS

    print("=" * 75)
    print("COMPONENT-BY-COMPONENT ABLATION -- load checkpoints + ensembling suite")
    print(f"device={DEVICE}  datasets={args.datasets}")
    print(f"out_root={args.out_root}")
    print(f"configs={[c['name'] for c in configs]}")
    print("=" * 75)

    summaries = {}
    all_rows = []
    for subset in args.datasets:
        s = run_dataset(subset, configs, data_root=args.data_root, out_root=args.out_root)
        if s is not None:
            summaries[subset] = s
            all_rows.extend(s['csv_rows'])

    combined = os.path.join(args.out_root, 'all_datasets_full_metrics_table.csv')
    if write_csv(combined, all_rows):
        print(f"\nwrote combined metrics table: {combined}")

    print(f"\n{'=' * 75}\nFINAL SUMMARY -- mean ensemble NASA\n{'=' * 75}")
    for subset, s in summaries.items():
        print(f"\n  {subset}:")
        for cfg in configs:
            entry = s['configs'].get(cfg['name'])
            if entry and entry['ensembles'].get('mean'):
                print(f"    {cfg['name']:<22}: {entry['ensembles']['mean']['nasa']:.2f}")
            else:
                print(f"    {cfg['name']:<22}: not found")
    print("=" * 75)
    return summaries


if __name__ == '__main__':
    main()
