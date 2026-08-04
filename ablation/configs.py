"""
The cumulative component ladder.

Each rung adds ONE component to the rung above it, so the difference between
two adjacent rows is attributable to that component alone. Nothing is ever
removed on the way up, which is what makes the ladder cumulative rather than
a leave-one-out study.

  tcn_baseline   dilated causal convolutions, 14 raw sensors. The reference:
                 no attention anywhere, so everything above has to beat the
                 cheaper and more conventional encoder.
  transformer    swaps the conv stack for a single full-causal attention path.
  near_far       + the second, local path and the content-conditioned gate
                 that mixes them, including the last block's near->far
                 cross-attention. That cross-attention belongs to the
                 dual-path design itself, so every rung above carries it too,
                 not only the full model.
  geometry       + the 4 Riemannian / Wasserstein drift features (18 total).
  sensor_branch  + the iTransformer-style cross-sensor branch and its
                 cross-attention fusion.
  full_model     + the reverse-time encoding and the healthy-window contrast.

The spectral gate is not a rung. An FFT band-energy gate was run over 5 seeds
on both subsets, in two placements, and never moved NASA/RMSE/MAE beyond seed
noise, so the ladder has nothing to say about it.

RUN_ORDER, not this list, decides what trains first: full_model runs first so
the headline numbers exist early, while the paired step-by-step deltas are
always computed down THIS list, in progressive-complexity order. Reordering
CONFIGS would silently change which pairs the delta table compares.

A rung's name is its checkpoint filename prefix (<name>_seed<seed>.pt), so
renaming a rung orphans its checkpoints.
"""

CONFIGS = [
    dict(name='tcn_baseline', backbone='tcn', use_geometry=False,
         use_sensor_branch=False, use_reverse=False, use_healthy=False),
    dict(name='transformer', backbone='far_only', use_geometry=False,
         use_sensor_branch=False, use_reverse=False, use_healthy=False),
    dict(name='near_far', backbone='both', use_geometry=False,
         use_sensor_branch=False, use_reverse=False, use_healthy=False),
    dict(name='geometry', backbone='both', use_geometry=True,
         use_sensor_branch=False, use_reverse=False, use_healthy=False),
    dict(name='sensor_branch', backbone='both', use_geometry=True,
         use_sensor_branch=True, use_reverse=False, use_healthy=False),
    dict(name='full_model', backbone='both', use_geometry=True,
         use_sensor_branch=True, use_reverse=True, use_healthy=True),
]

# Training order only. Analysis always walks CONFIGS.
RUN_ORDER = ['full_model', 'tcn_baseline', 'transformer', 'near_far',
             'geometry', 'sensor_branch']

SEEDS = [53, 51, 43, 42, 3]


def config_by_name(name):
    for cfg in CONFIGS:
        if cfg['name'] == name:
            return cfg
    raise KeyError(f"unknown config {name!r}; known: {[c['name'] for c in CONFIGS]}")


def configs_in_run_order(configs=None):
    """Orders configs for TRAINING: full_model first, then the staircase.

    Anything not named in RUN_ORDER keeps its position at the end, so a config
    added to CONFIGS still trains even if RUN_ORDER was not updated.
    """
    configs = list(configs if configs is not None else CONFIGS)
    rank = {name: i for i, name in enumerate(RUN_ORDER)}
    return sorted(configs, key=lambda c: rank.get(c['name'], len(RUN_ORDER)))
