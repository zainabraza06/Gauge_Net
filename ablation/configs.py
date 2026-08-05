"""
The cumulative component ladder.

Each rung adds ONE component to the rung above it, so the difference between
two adjacent rows is attributable to that component alone. Nothing is ever
removed on the way up, which is what makes the ladder cumulative rather than
a leave-one-out study.

  transformer    a single full-causal attention path over the 14 raw
                 sensors. The reference every rung above is measured from.
  near_far       + the second, local path and the content-conditioned gate
                 that mixes them, including the last block's near->far
                 cross-attention. That cross-attention belongs to the
                 dual-path design itself, so every rung above carries it too,
                 not only the full model.
  geometry       + the 4 Riemannian / Wasserstein drift features (18 total).
  sensor_branch  + the iTransformer-style cross-sensor branch and its
                 cross-attention fusion.
  full_model     + the reverse-time encoding and the healthy-window contrast.
  full_model_regime
                 + a learned embedding of the operating regime, added per
                 timestep. MULTI-REGIME SUBSETS ONLY (FD002, FD004): the
                 per-regime z-score removes each regime's mean, and with it
                 which flight condition a row was recorded in, so this hands
                 that back. On FD001/FD003 every row is one regime, so the
                 rung is skipped rather than trained on a constant.

Neither the TCN baseline nor the spectral gate is a rung. The convolutional
baseline was dropped: the study is about what the attention architecture's
components contribute, and a conv encoder differs from the transformer rungs
in too many ways at once for the step above it to attribute anything.

The spectral gate is not a rung either. An FFT band-energy gate was run over 5
seeds on both subsets, in two placements, and never moved NASA/RMSE/MAE beyond
seed noise, so the ladder has nothing to say about it.

RUN_ORDER, not this list, decides what trains first: full_model runs first so
the headline numbers exist early, while the paired step-by-step deltas are
always computed down THIS list, in progressive-complexity order. Reordering
CONFIGS would silently change which pairs the delta table compares.

A rung's name is its checkpoint filename prefix (<name>_seed<seed>.pt), so
renaming a rung orphans its checkpoints.
"""

CONFIGS = [
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
    dict(name='full_model_regime', backbone='both', use_geometry=True,
         use_sensor_branch=True, use_reverse=True, use_healthy=True,
         use_regime=True),
]

# Training order only. Analysis always walks CONFIGS. full_model_regime is
# last in both: it is the top of the staircase, so running it early would put
# a rung before the one it is measured against.
RUN_ORDER = ['full_model', 'transformer', 'near_far', 'geometry',
             'sensor_branch', 'full_model_regime']

SEEDS = [53, 51, 43, 42, 3]


def config_by_name(name):
    for cfg in CONFIGS:
        if cfg['name'] == name:
            return cfg
    raise KeyError(f"unknown config {name!r}; known: {[c['name'] for c in CONFIGS]}")


def applicable_configs(configs, subset_has_regimes):
    """Drops the rungs a subset cannot support.

    Only full_model_regime is conditional: on a single-condition subset its
    embedding would index one constant id, so it would be full_model with a
    few dead parameters and a rung name implying a comparison that was never
    made. Skipping is reported by the caller rather than done silently.
    """
    if subset_has_regimes:
        return list(configs), []
    keep = [c for c in configs if not c.get('use_regime')]
    dropped = [c['name'] for c in configs if c.get('use_regime')]
    return keep, dropped


def configs_in_run_order(configs=None):
    """Orders configs for TRAINING: full_model first, then the staircase.

    Anything not named in RUN_ORDER keeps its position at the end, so a config
    added to CONFIGS still trains even if RUN_ORDER was not updated.
    """
    configs = list(configs if configs is not None else CONFIGS)
    rank = {name: i for i, name in enumerate(RUN_ORDER)}
    return sorted(configs, key=lambda c: rank.get(c['name'], len(RUN_ORDER)))
