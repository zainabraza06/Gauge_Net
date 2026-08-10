"""
Builds a deployment bundle: everything inference needs that is NOT in a
checkpoint.

    python -m app.bundle FD001
    python -m app.bundle FD001 FD002 FD003 FD004

A trained checkpoint is only half of a deployed model. The other half is the
fitted state of the preprocessing: the normalisation statistics, the healthy
geometric reference, and the fleet median life that turns a cycle number into
a life fraction. All three were fitted on TRAIN, none are recoverable from the
weights, and without them an uploaded CSV cannot be put on the scale the model
was trained on.

So this stage reads the preprocessing output and freezes those three things
into one small file per subset:

    normalisation   global mu/sigma, or the regime clusterer + per-regime
                    mu/sigma for FD002/FD004
    geometry refs   the healthy correlation matrix and the healthy per-sensor
                    marginals, pooled over train engines' first HEALTHY_FRAC
    fml             median fleet life, from train only

plus the per-seed validation NASA scores, which the weighted ensembling
techniques need, and which subset reports which technique.

Everything here is fitted on TRAIN and nothing is recomputed at serve time, so
the app cannot accidentally re-fit a statistic on whatever the user uploaded.
"""

import argparse
import json
import os
import pickle

import numpy as np

from model.config import DATASETS, DATA_ROOT, HEALTHY_FRAC, W
from model.data import pipeline_path
from model.geometry import compute_ref_spd_single, compute_was_ref_single

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(REPO_ROOT, 'app', 'bundles')

# Where each subset's reported model lives, and how its seeds are combined.
# The ensembling techniques mirror ablation.figures.BEST_PICKS. FD004's weights
# are the FD002->FD004 transfer arm, which is why its files are named by seed
# alone: they were written by the transfer stage, not by the ladder.
CHECKPOINT_ROOT = 'checkpoints'
DEPLOY = {
    'FD001': dict(prefix='sensor_branch_linear_far_seed', ensemble='mean'),
    'FD002': dict(prefix='sensor_branch_linear_far_seed', ensemble='softmax_sharp'),
    'FD003': dict(prefix='sensor_branch_linear_far_seed', ensemble='softmax_soft'),
    'FD004': dict(prefix='seed', ensemble='softmax_sharp'),
}


def checkpoint_dir(subset):
    return os.path.join(REPO_ROOT, CHECKPOINT_ROOT, subset)


def val_nasa_by_seed(subset):
    """Per-seed validation NASA, for the weighted ensembling techniques.

    Read from whichever results file the run that produced the checkpoints
    wrote. Returns None when it cannot be found -- the app then falls back to
    a plain mean, which needs no scores, rather than inventing weights.
    """
    d = checkpoint_dir(subset)
    for name in ('results.json', f'{subset}_progressive_results.json'):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            raw = json.load(f)
        records = raw if isinstance(raw, list) else raw.get(
            'sensor_branch_linear_far', [])
        scores = {int(r['seed']): float(r['best_val_nasa'])
                  for r in records
                  if isinstance(r, dict) and 'seed' in r and 'best_val_nasa' in r}
        if scores:
            return scores
    return None


def recompute_val_nasa(subset, ckpts, tr_geom, traj_fn, fml, device=None):
    """Validation NASA per seed, re-measured from the checkpoints.

    A fallback for runs whose results file did not keep the settled rung's
    per-seed scores. The split is reconstructed from the seed alone, and the
    snapshots from the seed alone, so this measures the same engines at the
    same truncations the training run early-stopped on.

    It is NOT the recorded number: MC dropout is stochastic and the recorded
    score came from the best epoch's RNG state. It is close enough to weight
    an ensemble -- and much closer to the reported configuration than dropping
    to an unweighted mean, which is the only other option. The bundle records
    that these were recomputed so a reader is not told otherwise.
    """
    import torch
    from model.config import SCALE, MC_PASSES_VAL
    from model.gaugenet import build_model
    from model.geometry import GEOM_COLS
    from model.inference import mc_predict
    from model.metrics import nasa_score
    from model.windows import build_val_snapshots, val_engines_for_seed
    from ablation.configs import config_by_name

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = config_by_name('sensor_branch_linear_far')
    fn = list(traj_fn) + list(GEOM_COLS)

    scores = {}
    for seed, path in ckpts.items():
        val_eng = val_engines_for_seed(tr_geom, seed)
        df_va = tr_geom[tr_geom['unit'].isin(val_eng)]
        win, rul, lf = build_val_snapshots(df_va, fn, fml, seed)
        model = build_model(len(fn), fml, cfg, device=device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        with torch.no_grad():
            p = mc_predict(model, win.to(device), lf.to(device), M=MC_PASSES_VAL)
        scores[seed] = nasa_score(p.cpu().numpy() * SCALE, rul.numpy())
        del model
    return scores


def cached_train_geometry(subset):
    """The train frame with its geometry columns, from the run's own cache.

    Read directly rather than through build_geometry: these caches predate the
    signature stamp, so the normal path would treat them as stale and spend
    longer than a training run recomputing features that are already correct.
    Only used to rebuild validation splits, never to serve.
    """
    path = os.path.join(checkpoint_dir(subset), 'geom_cache.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f).get('tr_ext')


def discover_checkpoints(subset):
    """The checkpoint files for one subset, keyed by seed."""
    spec = DEPLOY[subset]
    d = checkpoint_dir(subset)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"No checkpoint directory for {subset} at {d}")
    found = {}
    for fname in os.listdir(d):
        if not (fname.startswith(spec['prefix']) and fname.endswith('.pt')):
            continue
        try:
            found[int(fname[len(spec['prefix']):-3])] = os.path.join(d, fname)
        except ValueError:
            continue
    if not found:
        raise FileNotFoundError(
            f"No '{spec['prefix']}*.pt' checkpoints for {subset} in {d}")
    return dict(sorted(found.items()))


def read_normalisation(out, subset, traj_fn):
    """The fitted normalisation, from either pipeline layout.

    The current pipeline writes `normalization_stats[subset]` carrying a
    'scheme' field. Older outputs wrote the global statistics under the same
    key without a scheme, and the per-regime statistics under a top-level
    `regime_info` instead. The fields themselves are identical, so rather than
    require the pickles be regenerated, both layouts are read and normalised
    into the one shape the serving path expects.

    The scheme is inferred from WHICH statistics are present rather than
    assumed from the subset name -- a subset normalised the wrong way is
    exactly the failure this should catch, not reproduce.
    """
    stats = (out.get('normalization_stats') or {}).get(subset)
    if stats and 'scheme' in stats:
        return stats

    if stats and 'trajectory_mu' in stats:
        return dict(scheme='global',
                    trajectory_mu=np.asarray(stats['trajectory_mu']),
                    trajectory_sig=np.asarray(stats['trajectory_sig']),
                    sensor_order=list(stats.get('sensor_order', traj_fn)))

    regime = out.get('regime_info')
    if isinstance(regime, dict) and subset in regime:
        regime = regime[subset]
    if isinstance(regime, dict) and 'kmeans_centers' in regime:
        return dict(scheme='per-regime', n_regimes=int(regime['n_regimes']),
                    op_scaler_mean=np.asarray(regime['op_scaler_mean']),
                    op_scaler_scale=np.asarray(regime['op_scaler_scale']),
                    kmeans_centers=np.asarray(regime['kmeans_centers']),
                    regime_mu={int(k): np.asarray(v)
                               for k, v in regime['regime_mu'].items()},
                    regime_sig={int(k): np.asarray(v)
                                for k, v in regime['regime_sig'].items()},
                    sensor_order=list(regime.get('sensor_order', traj_fn)))

    raise KeyError(
        f"{subset}: the pipeline output carries neither global normalisation "
        f"statistics nor regime_info, so there is nothing to put an upload on "
        f"the training scale with. Regenerate it: python -m preprocessing {subset}")


def build(subset, data_root=None, out_dir=BUNDLE_DIR):
    pkl = pipeline_path(subset, data_root or DATA_ROOT)
    if not os.path.exists(pkl):
        raise FileNotFoundError(
            f"No preprocessed input at {pkl}\n\n"
            f"The bundle freezes statistics that were fitted on the TRAIN split, so it\n"
            f"has to be built where the C-MAPSS data is. Run:\n"
            f"    python -m preprocessing {subset}\n"
            f"then re-run this command.")

    with open(pkl, 'rb') as f:
        out = pickle.load(f)
    traj_fn = out['trajectory_feature_names'][subset]
    tr = out['feature_data'][subset]['train']
    norm = read_normalisation(out, subset, traj_fn)

    # Fitted on the same normalised train frame the ladder trained on, so the
    # reference an uploaded engine is measured against is the one the weights
    # were learned against.
    ref_spd = compute_ref_spd_single(tr, traj_fn)
    was_ref = compute_was_ref_single(tr, traj_fn)
    fml = float(np.median(tr.groupby('unit')['cycle'].max().values))

    ckpts = discover_checkpoints(subset)
    val_nasa, val_source = val_nasa_by_seed(subset), 'recorded at training time'
    if val_nasa is None or set(ckpts) - set(val_nasa):
        tr_geom = cached_train_geometry(subset)
        if tr_geom is not None:
            print(f"  {subset}: validation scores missing for the settled rung -- "
                  f"re-measuring them from the checkpoints")
            val_nasa = recompute_val_nasa(subset, ckpts, tr_geom, traj_fn, fml)
            val_source = 'recomputed from the checkpoints (not the recorded scores)'
        elif val_nasa is None:
            print(f"  {subset}: no per-seed validation NASA and no geometry cache to "
                  f"re-measure with -- the app will fall back to a plain mean ensemble")
            val_source = None

    bundle = dict(
        subset=subset, traj_fn=list(traj_fn), window=W, healthy_frac=HEALTHY_FRAC,
        normalization=norm, ref_spd=ref_spd, was_ref=was_ref, fml=fml,
        ensemble=DEPLOY[subset]['ensemble'], val_nasa=val_nasa,
        val_nasa_source=val_source,
        # Forward slashes always. os.path.relpath emits the host separator, so
        # a bundle built on Windows would carry 'checkpoints\FD002\...' -- and
        # on Linux a backslash is an ordinary filename character, not a
        # separator, so every checkpoint would be reported missing on the
        # deployment target. A bundle is a portable artefact; it stores a
        # portable path.
        checkpoints={s: os.path.relpath(p, REPO_ROOT).replace(os.sep, '/')
                     for s, p in ckpts.items()})

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{subset}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    size_kb = os.path.getsize(path) // 1024
    print(f"  {subset}: {len(ckpts)} checkpoints, scheme={norm['scheme']}, "
          f"fml={fml:.0f}, ensemble={bundle['ensemble']} -> {path} ({size_kb} KB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('subsets', nargs='*', default=DATASETS)
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--out-dir', default=BUNDLE_DIR)
    args = ap.parse_args()

    print("=" * 70)
    print("DEPLOYMENT BUNDLES -- freezing train-fitted preprocessing state")
    print("=" * 70)
    for subset in args.subsets:
        build(subset, args.data_root, args.out_dir)


if __name__ == '__main__':
    main()
