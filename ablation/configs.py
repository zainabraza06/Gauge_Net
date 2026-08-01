"""
The cumulative component ladder.

Each rung adds ONE component to the rung above it, so the difference between
two adjacent rows is attributable to that component alone. Nothing is ever
removed on the way up, which is what makes the ladder cumulative rather than
a leave-one-out study.

  1_backbone_only        single full-causal transformer path. The reference.
  2_plus_sensor_gate     + cross-sensor input gating
  3c_near_far_both       + the dual near/far paths with a content-conditioned
                           mixing gate (replaces the single path)
  4_plus_reverse         + the time-reversed encoding, learned blend
  5_plus_spectral        + frequency-content gating of the time features
  6_plus_healthy_FINAL   + the healthy-window contrast. The full model.

The rung name is also the checkpoint filename prefix
(<name>_seed<seed>.pt), so renaming a rung orphans its checkpoints.
'3c_near_far_both' keeps its original name for exactly that reason.
"""

CONFIGS = [
    dict(name='1_backbone_only', use_sensor_gate=False, path_mode='backbone',
         use_reverse=False, use_spectral=False, use_healthy=False),
    dict(name='2_plus_sensor_gate', use_sensor_gate=True, path_mode='backbone',
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

SEEDS = [51, 53, 43, 42, 3]


def config_by_name(name):
    for cfg in CONFIGS:
        if cfg['name'] == name:
            return cfg
    raise KeyError(f"unknown config {name!r}; known: {[c['name'] for c in CONFIGS]}")
