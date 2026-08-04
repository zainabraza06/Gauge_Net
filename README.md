# IGRFNet — geometry-informed RUL prediction on C-MAPSS

Component-by-component ablation of a transformer-based Remaining Useful Life
(RUL) estimator on the NASA C-MAPSS turbofan datasets **FD001** and **FD003**.

Every stage is causal: a prediction at cycle *t* is computed from cycles
≤ *t* of that engine alone, and every normalisation statistic is fitted on
the training split only.

---

## Layout

```
preprocessing/     raw C-MAPSS text  ->  leak-free feature pickle
  pipeline.py        z-score, RUL cap, trajectory + label only
  __main__.py        python -m preprocessing FD001

model/             everything shared by training and evaluation
  config.py          hyperparameters and paths (env-overridable)
  metrics.py         NASA score, RMSE, MAE, mean error, Accuracy(±k)
  losses.py          unweighted asymmetric Huber
  geometry.py        Riemannian / Wasserstein drift features (cached)
  windows.py         causal windowing, dataset, validation snapshots
  modules.py         sensor gate, temporal blocks, pooling, healthy contrast
  igrfnet.py         the ablatable network + build_model(cfg)
  data.py            load a subset and attach geometry features
  inference.py       MC-dropout prediction, per-engine scoring, checkpoints

tests/
  smoke_test.py      end-to-end run on synthetic data, no NASA archive needed

ablation/          the study itself
  configs.py         the six-rung cumulative ladder
  train.py           python -m ablation.train
  ensembling.py      eight ways of combining a config's seeds
  evaluate.py        python -m ablation.evaluate
```

## Quick start

```bash
git clone <this-repo> && cd Research
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download **CMAPSSData** from the
[NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and put six files in `data/`:

```
data/
  train_FD001.txt  test_FD001.txt  RUL_FD001.txt
  train_FD003.txt  test_FD003.txt  RUL_FD003.txt
```

Then run the three stages from the repository root:

```bash
python -m preprocessing FD001        # -> data/pipeline_output/output_FD001_final.pkl
python -m preprocessing FD003        # -> data/pipeline_output/output_FD003_final.pkl

python -m ablation.train             # full ladder, all seeds
python -m ablation.evaluate          # score checkpoints + ensemble
```

Nothing needs configuring: `data/` and `results/` default to directories
inside the clone, and both are git-ignored. To put them elsewhere, pass
`--data-root` / `--out-root` to any command, or set `CMAPPS_DATA_ROOT` /
`CMAPPS_OUT_ROOT`.

To check the install before downloading anything, run the smoke test — it
generates synthetic C-MAPSS-format data and drives all three stages:

```bash
python tests/smoke_test.py
```

**A full run needs a GPU.** The ladder is 6 configs x 5 seeds x 2 datasets =
60 trainings, and `ablation.train` warns if no CUDA device is visible. To
sanity-check the pipeline on CPU first, run one rung with one seed:

```bash
python -m ablation.train --datasets FD001 --configs 6_plus_healthy_FINAL --seeds 42
python -m ablation.evaluate --datasets FD001 --configs 6_plus_healthy_FINAL
```

The first run of a dataset also computes the geometry features, which takes
longer than a training run; they are cached in
`results/component_ablation_v2/<dataset>/geom_cache.pkl` and reused by every
later run.

Training resumes: a `(config, seed)` whose checkpoint already exists is
skipped unless `--overwrite` is passed. Evaluation discovers whichever
`<config>_seed*.pt` files exist, so an interrupted run is reported on exactly
as far as it got.

## The ablation ladder

Each rung adds **one** component to the rung above, so any two adjacent rows
differ by that component alone.

| Rung | Adds | Params (18 features) |
|---|---|---|
| `1_backbone_only` | single full-causal transformer path | 309,218 |
| `2_plus_sensor_gate` | cross-sensor input gating | 310,675 |
| `3c_near_far_both` | dual near/far paths + content-conditioned mixing gate | 572,497 |
| `4_plus_reverse` | time-reversed encoding, learned blend | 722,066 |
| `6_plus_healthy_FINAL` | healthy-window contrast | 768,722 |

A rung's name is its checkpoint filename prefix (`<name>_seed<seed>.pt`), so
renaming one orphans its checkpoints.

## Features and objective

**18 features**: 14 raw sensors + 4 geometry columns. The 14 causal
expanding-baseline columns were ablated against a no-baseline control and made
TEST_NASA worse on both subsets, most likely because they are redundant with
the healthy-window cross-attention, which already performs a learned
per-engine offset comparison. `baseline_feature_names` is emitted as an empty
list so loaders keep working.

**No sample weighting.** The loss is a plain unweighted asymmetric Huber,
whose late side (over-predicting RUL) is scaled by `BETA = 1.5`. Two weighted
variants were ablated against it — a smooth ramp in capped RUL, and an added
differentiable NASA-score surrogate — and neither beat it, so both are gone
rather than switched off.

Weighting is also where a real bug lived, worth recording: FD001 used to emit
`onset`/`T_i`/`phase` and FD003 did not, and the dataset silently substituted
`onset=0, phase=1`, so FD003 ramped its weight across the whole trajectory
while FD001 ramped it only across the degradation window — two objectives from
code that looked dataset-agnostic. The pipeline now emits 14 z-scored sensors
and the capped RUL and nothing else, and both the pipeline's verification and
the smoke test assert that schema is closed.

Results go to `component_ablation_v2/`. Checkpoints under the older
`component_ablation/` are **not** comparable: different loss, different
feature count.

## Geometry features

Four columns per row, measuring how far a sliding window has drifted from a
healthy reference — against the **fleet** (`riem_global`, `was_global`) and
against **the engine's own history** (`riem_personal`, `was_personal`). The
Riemannian pair responds to changes in coupling *between* sensors, the
Wasserstein pair to changes in each sensor's own distribution. They are
cached in `geom_cache.pkl`, since computing them costs more than training.

## Ensembling

The seeds of one config are combined eight ways: `mean`, `median`,
`best_k_mean`, `val_weighted`, `softmax_sharp`, `softmax_soft`,
`rank_average`, `bias_corrected`. A technique that cannot be computed from
the members found is skipped **with a recorded reason**, so a missing row in
the results table is never ambiguous.

Two things worth carrying into any write-up:

- **Validation-score weighting is only roughly comparable.** Each seed holds
  out different engines, so `best_val_nasa` carries split difficulty as well
  as model quality. It ranks members adequately; it is not a calibrated
  quality estimate.
- **Ridge stacking is not offered.** Fitting per-member blend weights needs a
  snapshot set every member can be scored on and none of them trained on —
  the engines held out by *all* members. With five seeds holding out 20% each
  that intersection is empty, so the technique cannot be fitted on this
  design without either leaking a member's training data into its own weight
  or misaligning the columns. Any `ridge_stacking` number from an earlier run
  came from a misaligned fit and should not be reported.

## Metrics

`nasa` (PHM08 asymmetric score, a **sum** — only comparable across equally
sized test sets), `rmse`, `mae`, `mean_error` (signed; positive means
over-predicting RUL, the dangerous direction) and `Accuracy(±5/10/13/15/20)`.

Model selection during training is on validation NASA, not RMSE: the two
disagree about the best epoch because NASA penalises late predictions far
harder.

## Outputs

Per dataset, under `results/component_ablation_v2/<dataset>/`:

| File | Contents |
|---|---|
| `<config>_seed<seed>.pt` | selected checkpoint for one member |
| `geom_cache.pkl` | cached geometry features, reused by every run |
| `<dataset>_training_summary.json` | per-config test NASA mean/std |
| `loaded_component_ensembles_full.json` | every metric plus skip reasons |
| `<dataset>_full_metrics_table.csv` | one row per (config, technique) |

and `all_datasets_full_metrics_table.csv` at the output root.

## Requirements

Python 3.10+, `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`.
