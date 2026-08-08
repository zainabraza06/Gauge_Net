"""
End-to-end smoke test on synthetic data.

    python tests/smoke_test.py

Generates C-MAPSS-format trajectories, then runs preprocessing -> training ->
evaluation exactly as a reviewer would, in a scratch directory. It checks that
the plumbing works -- not that the model is any good, since it trains for two
epochs on invented sensors. Use it to confirm an install before downloading
the real archive.

Takes a few minutes on CPU. Rerunning reuses the scratch workspace, so the
stages that already completed are skipped.
"""

import os
import pickle
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(REPO, '.smoke_workspace')
DATA = os.path.join(WS, 'data')
OUT = os.path.join(WS, 'results')
CONFIG = 'sensor_branch_linear_far'
# A second, cheap rung so the paired step-delta table has a pair to compare.
# Two seeds because the deltas need >=2 shared seeds to be computable at all.
BASELINE_CONFIG = 'transformer'
SEEDS = [42, 43]
SEED = SEEDS[0]
SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
           's13', 's14', 's15', 's17', 's20', 's21']

rng = np.random.default_rng(0)


# Six discrete flight conditions, the shape FD002/FD004 actually have.
REGIMES = [(0.0, 0.0, 100.0), (10.0, 0.25, 100.0), (20.0, 0.70, 100.0),
           (25.0, 0.62, 60.0), (35.0, 0.84, 100.0), (42.0, 0.84, 100.0)]
REGIME_OFFSET = 40.0     # how far a regime shifts every sensor


def make_engine(uid, life, multi_regime=False):
    """A degrading trajectory: sensors drift on a quadratic ramp toward failure.

    With multi_regime, each cycle is flown in one of six conditions and every
    sensor is offset by a per-regime constant far larger than the degradation
    signal -- which is exactly why those subsets cannot use a single z-score.
    """
    rows = []
    for cycle in range(1, life + 1):
        drift = (cycle / life) ** 2
        r = int(rng.integers(0, len(REGIMES))) if multi_regime else None
        op = REGIMES[r] if multi_regime else (0.0, 0.0, 100.0)
        offset = REGIME_OFFSET * r if multi_regime else 0.0
        sensors = [500.0 + s * 13.0 + offset + (30.0 if s % 3 == 0 else -20.0) * drift
                   + rng.normal(0, 0.6) for s in range(21)]
        rows.append([uid, cycle] + list(op) + sensors)
    return rows


def write_subset(subset, n_train=14, n_test=9, multi_regime=False):
    """Writes train/test/RUL files in the exact whitespace format of C-MAPSS."""
    train = []
    for u in range(1, n_train + 1):
        train += make_engine(u, int(rng.integers(150, 260)), multi_regime)

    test, ruls = [], []
    for u in range(1, n_test + 1):
        life = int(rng.integers(150, 260))
        cut = int(rng.integers(90, life - 20))     # test engines stop mid-life
        test += make_engine(u, cut, multi_regime)
        ruls.append(min(life - cut, 125))

    def fmt(rows):
        return "\n".join(" ".join(f"{v:.4f}" if i >= 2 else str(int(v))
                                  for i, v in enumerate(r)) for r in rows)

    os.makedirs(DATA, exist_ok=True)
    for name, body in [(f'train_{subset}.txt', fmt(train)),
                       (f'test_{subset}.txt', fmt(test)),
                       (f'RUL_{subset}.txt', "\n".join(str(r) for r in ruls))]:
        with open(os.path.join(DATA, name), 'w') as f:
            f.write(body + "\n")


def run(args, expect_fail=False):
    print(f"\n$ python {' '.join(args)}")
    p = subprocess.run([sys.executable] + args, cwd=REPO, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if (p.returncode != 0) != expect_fail:
        print(out[-3000:])
        raise SystemExit(f"FAILED: unexpected exit code {p.returncode}")
    return out


def write_stub(name, body):
    """A driver that shortens the run: two epochs, three MC passes. The MC
    default is bound at definition time, so it is patched on the function."""
    path = os.path.join(WS, name)
    with open(path, 'w') as f:
        f.write(f"import sys; sys.path.insert(0, r'{REPO}')\n"
                f"import model.inference as I\n"
                f"I.predict_engine.__defaults__ = ('cpu', 3, 9)\n" + body)
    return path


def main():
    os.makedirs(WS, exist_ok=True)
    print("=" * 70)
    print("SMOKE TEST -- synthetic data through preprocessing -> train -> evaluate")
    print(f"workspace: {WS}")
    print("=" * 70)

    if not os.path.exists(os.path.join(DATA, 'train_FD001.txt')):
        out = run(['-m', 'preprocessing', 'FD001', '--data-root', DATA], expect_fail=True)
        assert 'Missing 3 input file(s)' in out and 'nasa.gov' in out
        print("  [1] missing inputs -> error naming each file and the download URL  OK")
        write_subset('FD001')
    else:
        print("  [1] (synthetic data already present)")

    out = run(['-m', 'preprocessing', 'FD001', '--data-root', DATA])
    assert 'ALL CHECKS PASSED' in out, out[-2000:]
    assert '14 trajectory features, 0 baseline features' in out
    pkl = os.path.join(DATA, 'pipeline_output', 'output_FD001_final.pkl')
    assert os.path.exists(pkl)
    # The pickle must carry the trajectory features and the label, nothing
    # else: a stray sample_weight/onset/T_i/phase column is how the two
    # subsets previously drifted onto different objectives.
    with open(pkl, 'rb') as f:
        cols = set(pickle.load(f)['feature_data']['FD001']['train'].columns)
    assert not cols - ({'unit', 'cycle', 'rul_true', 'rul'} | set(SENSORS)), sorted(cols)
    print("  [2] preprocessing -> output_FD001_final.pkl, closed schema, checks passed  OK")

    # FD002 covers the multi-regime preprocessing path. The staircase itself
    # is identical on every subset, so it is exercised on FD001 alone.
    if not os.path.exists(os.path.join(DATA, 'train_FD002.txt')):
        write_subset('FD002', multi_regime=True)
    out = run(['-m', 'preprocessing', 'FD002', '--data-root', DATA])
    assert 'ALL CHECKS PASSED' in out, out[-2500:]
    assert 'per-regime, KMeans(k=6)' in out
    assert 'every regime populated in train: True' in out
    assert 'no test rows stranded in an unfitted regime: True' in out
    with open(os.path.join(DATA, 'pipeline_output', 'output_FD002_final.pkl'), 'rb') as f:
        fd002 = pickle.load(f)
    # Per-regime z-scoring must remove the regime offsets: a single global
    # z-score would leave a spread far wider than 1 because the offsets
    # dominate the degradation signal.
    vals = fd002['feature_data']['FD002']['train'][SENSORS].values
    assert abs(vals.mean()) < 0.05 and abs(vals.std() - 1.0) < 0.05, (vals.mean(), vals.std())
    assert fd002['normalization_stats']['FD002']['scheme'] == 'per-regime'
    print("  [2b] FD002 -> 6 regimes fitted on train, offsets normalised away  OK")

    if not os.path.isdir(os.path.join(OUT, 'FD001')):
        out = run(['-m', 'ablation.evaluate', '--datasets', 'FD001',
                   '--data-root', DATA, '--out-root', OUT])
        assert 'no checkpoint directory' in out
        print("  [3] evaluate before training -> clean skip, says to train first  OK")

    train_stub = write_stub('run_train.py', f"""
import ablation.train as T
T.EPOCHS = 2; T.WARMUP_EP = 1; T.PATIENCE = 2
T.run_dataset('FD001', [T.config_by_name(n) for n in ({CONFIG!r}, {BASELINE_CONFIG!r})], {SEEDS!r},
              data_root=r'{DATA}', out_root=r'{OUT}')
""")
    out = run([train_stub])
    assert 'TEST_NASA' in out, out[-2000:]
    assert 'features: 14 sensors + 4 geometry available = 18' in out
    assert 'sensor_branch_linear_far  (18 features, geometry=True, far_attn=linear)' in out
    # transformer trains on the 14 sensors alone: the geometry rung is above
    # it on the ladder, so a config below it must not be fed those columns.
    assert 'transformer  (14 features, geometry=False, far_attn=full)' in out
    for name in (CONFIG, BASELINE_CONFIG):
        for seed in SEEDS:
            assert os.path.exists(os.path.join(OUT, 'FD001', f'{name}_seed{seed}.pt'))
        assert os.path.exists(os.path.join(OUT, 'FD001', f'{name}_preds.npz'))
    print("  [4] training -> checkpoints + raw predictions, per-rung features correct  OK")

    out = run([train_stub])
    assert 'checkpoint exists, skipping' in out
    print("  [5] rerun -> existing checkpoint skipped, so runs resume  OK")

    eval_stub = write_stub('run_eval.py', f"""
import ablation.evaluate as E
E.run_dataset('FD001', [E.config_by_name(n) for n in ({BASELINE_CONFIG!r}, {CONFIG!r})],
              data_root=r'{DATA}', out_root=r'{OUT}')
""")
    out = run([eval_stub])
    assert f'found seeds: {SEEDS}' in out, out[-2000:]
    # Two members: trimmed_mean needs three, so it must skip WITH a reason,
    # while rank_average now has enough members to be computed.
    assert 'trimmed_mean    : SKIPPED (needs >=3 members' in out
    csv = os.path.join(OUT, 'FD001', 'FD001_full_metrics_table.csv')
    assert os.path.exists(csv)
    print("  [6] evaluate -> scored checkpoints, member-count skips correct  OK")

    # Techniques are tested against the mean per ENGINE, which is where the
    # study actually has power -- unlike the five-seed rung deltas.
    assert "technique vs 'mean', paired per engine" in out, out[-3000:]
    for name in (CONFIG, BASELINE_CONFIG):
        assert os.path.exists(os.path.join(OUT, 'FD001', f'{name}_ensemble_preds.npz'))

    assert f'{BASELINE_CONFIG} -> {CONFIG}' in out, out[-3000:]
    assert 'boot 95% CI=' in out and 'paired-t p=' in out
    print("  [7] step deltas -> adjacent rungs paired by seed, tests reported  OK")

    assert os.path.exists(os.path.join(OUT, 'FD001', 'geom_cache.pkl'))
    assert 'recomputing' not in out
    print("  [8] geometry cache reused rather than recomputed  OK")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED -- the clone-and-run path works end to end")
    print(f"delete {WS} to reset")
    print("=" * 70)


if __name__ == '__main__':
    main()
