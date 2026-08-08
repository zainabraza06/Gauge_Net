"""
The cumulative component ladder.

Each rung adds ONE component to the rung above it, so the difference between
two adjacent rows is attributable to that component alone. Nothing is ever
removed on the way up, which is what makes the ladder cumulative rather than
a leave-one-out study.

  transformer               a single full-causal attention path over the 14
                            raw sensors. The reference every rung is measured
                            from.
  near_far                  + the second, local path and the content-
                            conditioned gate that mixes them, including the
                            last block's near->far cross-attention. That
                            cross-attention belongs to the dual-path design
                            itself, so every rung above carries it too.
  geometry                  + the 4 Riemannian / Wasserstein drift features
                            (18 features in total).
  sensor_branch             + the iTransformer-style cross-sensor branch and
                            its cross-attention fusion.
  sensor_branch_linear_far  the far path swapped from exact quadratic
                            attention to the O(n) linear approximation. The
                            settled model.

The rungs that are NOT here were all tried and dropped, and the reasons are
worth keeping because each is a different kind of negative result:

  tcn_baseline         a conv encoder differs from the transformer rungs in
                       too many ways at once for the step above it to
                       attribute anything to one component.
  spectral gate        an FFT band-energy gate never moved NASA/RMSE/MAE
                       beyond seed noise, over 5 seeds and two placements.
  reverse-time path    doubled the temporal parameters and did not survive as
                       the ladder was rebuilt around the sensor branch.
  healthy-window       the sensor branch subsumes it: both answer "how does
                       this engine differ from healthy", and the branch does
                       it from the window alone.
  regime conditioning  an embedding of the flight regime, on FD002/FD004
                       only. A rung every dataset cannot walk is not a rung;
                       the staircase is the same five steps on all four
                       subsets, and the regimes are handled where they arise,
                       in the per-regime normalisation.

RUN_ORDER, not this list, decides what trains first. The paired step-by-step
deltas are always computed down THIS list, in progressive-complexity order --
reordering CONFIGS would silently change which pairs the delta table compares.

A rung's name is its checkpoint filename prefix (<name>_seed<seed>.pt), so
renaming a rung orphans its checkpoints.
"""

CONFIGS = [
    dict(name='transformer', backbone='far_only', use_geometry=False,
         use_sensor_branch=False, far_attn_type='full'),
    dict(name='near_far', backbone='both', use_geometry=False,
         use_sensor_branch=False, far_attn_type='full'),
    dict(name='geometry', backbone='both', use_geometry=True,
         use_sensor_branch=False, far_attn_type='full'),
    dict(name='sensor_branch', backbone='both', use_geometry=True,
         use_sensor_branch=True, far_attn_type='full'),
    dict(name='sensor_branch_linear_far', backbone='both', use_geometry=True,
         use_sensor_branch=True, far_attn_type='linear'),
]

# Training order only. Analysis always walks CONFIGS. The settled model runs
# first so its numbers exist early, then the staircase fills in beneath it.
RUN_ORDER = ['sensor_branch_linear_far', 'transformer', 'near_far', 'geometry',
             'sensor_branch']

SEEDS = [53, 51, 43, 42, 3]


def config_by_name(name):
    for cfg in CONFIGS:
        if cfg['name'] == name:
            return cfg
    raise KeyError(f"unknown config {name!r}; known: {[c['name'] for c in CONFIGS]}")


def configs_in_run_order(configs=None):
    """Orders configs for TRAINING: the settled model first, then the rest.

    Anything not named in RUN_ORDER keeps its position at the end, so a config
    added to CONFIGS still trains even if RUN_ORDER was not updated.
    """
    configs = list(configs if configs is not None else CONFIGS)
    rank = {name: i for i, name in enumerate(RUN_ORDER)}
    return sorted(configs, key=lambda c: rank.get(c['name'], len(RUN_ORDER)))
