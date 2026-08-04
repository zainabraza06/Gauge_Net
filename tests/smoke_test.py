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
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(REPO, '.smoke_workspace')
DATA = os.path.join(WS, 'data')
OUT = os.path.join(WS, 'results')
CONFIG = '6_plus_healthy_FINAL'
SEED = 42

rng = np.random.default_rng(0)


def make_engine(uid, life):
    """A degrading trajectory: sensors drift on a quadratic ramp toward failure."""
    rows = []
    for cycle in range(1, life + 1):
        drift = (cycle / life) ** 2
        sensors = [500.0 + s * 13.0 + (30.0 if s % 3 == 0 else -20.0) * drift
                   + rng.normal(0, 0.6) for s in range(21)]
        rows.append([uid, cycle, 0.0, 0.0, 100.0] + sensors)
    return rows


def write_subset(subset, n_train=14, n_test=9):
    """Writes train/test/RUL files in the exact whitespace format of C-MAPSS."""
    train = []
    for u in range(1, n_train + 1):
        train += make_engine(u, int(rng.integers(150, 260)))

    test, ruls = [], []
    for u in range(1, n_test + 1):
        life = int(rng.integers(150, 260))
        cut = int(rng.integers(90, life - 20))     # test engines stop mid-life
        test += make_engine(u, cut)
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
    assert os.path.exists(os.path.join(DATA, 'pipeline_output', 'output_FD001_final.pkl'))
    print("  [2] preprocessing -> output_FD001_final.pkl, verification passed  OK")

    if not os.path.isdir(os.path.join(OUT, 'FD001')):
        out = run(['-m', 'ablation.evaluate', '--datasets', 'FD001',
                   '--data-root', DATA, '--out-root', OUT])
        assert 'no checkpoint directory' in out
        print("  [3] evaluate before training -> clean skip, says to train first  OK")

    train_stub = write_stub('run_train.py', f"""
import ablation.train as T
T.EPOCHS = 2; T.WARMUP_EP = 1; T.PATIENCE = 2
T.run_dataset('FD001', [T.config_by_name({CONFIG!r})], [{SEED}],
              data_root=r'{DATA}', out_root=r'{OUT}')
""")
    out = run([train_stub])
    assert 'TEST_NASA' in out, out[-2000:]
    assert 'features: 14 sensors + 0 baseline + 4 geometry = 18' in out
    assert os.path.exists(os.path.join(OUT, 'FD001', f'{CONFIG}_seed{SEED}.pt'))
    print("  [4] training -> checkpoint written, 18 features confirmed  OK")

    out = run([train_stub])
    assert 'checkpoint exists, skipping' in out
    print("  [5] rerun -> existing checkpoint skipped, so runs resume  OK")

    eval_stub = write_stub('run_eval.py', f"""
import ablation.evaluate as E
E.run_dataset('FD001', [E.config_by_name({CONFIG!r})],
              data_root=r'{DATA}', out_root=r'{OUT}')
""")
    out = run([eval_stub])
    assert f'found seeds: [{SEED}]' in out, out[-2000:]
    # One member: the techniques that need 2 or 3 must skip, with a reason.
    assert 'best_k_mean     : SKIPPED (needs >=3 members' in out
    assert 'rank_average    : SKIPPED (needs >=2 members' in out
    csv = os.path.join(OUT, 'FD001', 'FD001_full_metrics_table.csv')
    assert os.path.exists(csv)
    print("  [6] evaluate -> scored checkpoint, single-member skips correct  OK")

    assert os.path.exists(os.path.join(OUT, 'FD001', 'geom_cache.pkl'))
    assert 'recomputing' not in out
    print("  [7] geometry cache reused rather than recomputed  OK")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED -- the clone-and-run path works end to end")
    print(f"delete {WS} to reset")
    print("=" * 70)


if __name__ == '__main__':
    main()
