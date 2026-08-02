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
  pipeline.py        z-score, causal expanding baseline, RUL cap, onset/phase
  __main__.py        python -m preprocessing FD001

model/             everything shared by training and evaluation
  config.py          hyperparameters and paths (env-overridable)
  metrics.py         NASA score, RMSE, MAE, mean error, Accuracy(±k)
  losses.py          asymmetric Huber + degradation-phase weighting
  geometry.py        Riemannian / Wasserstein drift features (cached)
  windows.py         causal windowing, dataset, validation snapshots
  modules.py         sensor gate, spectral gate, temporal blocks, pooling
  igrfnet.py         the ablatable network + build_model(cfg)
  data.py            load a subset and attach geometry features
  inference.py       MC-dropout prediction, per-engine scoring, checkpoints

ablation/          the study itself
  configs.py         the six-rung cumulative ladder
  train.py           python -m ablation.train
  ensembling.py      nine ways of combining a config's seeds
  evaluate.py        python -m ablation.evaluate
```

## Data

Download **CMAPSSData** from the
[NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
so that `$CMAPPS_DATA_ROOT` contains `train_FD001.txt`, `test_FD001.txt`,
`RUL_FD001.txt` and the FD003 equivalents.

## Running

```bash
export CMAPPS_DATA_ROOT=/path/to/cmapps        # raw txt + pipeline_output/
export CMAPPS_OUT_ROOT=/path/to/checkpoints    # default: $CMAPPS_DATA_ROOT/component_ablation

python -m preprocessing FD001        # -> output_FD001_v2.pkl
python -m preprocessing FD003        # -> output_FD003_v5.pkl

python -m ablation.train                        # full ladder, all seeds
python -m ablation.train --datasets FD001 --configs 6_plus_healthy_FINAL --seeds 51
python -m ablation.evaluate                     # score checkpoints + ensemble
```

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
| `2_plus_sensor_gate` | cross-sensor input gating | 309,691 |
| `3c_near_far_both` | dual near/far paths + content-conditioned mixing gate | 571,513 |
| `4_plus_reverse` | time-reversed encoding, learned blend | 721,082 |
| `5_plus_spectral` | frequency-content gating | 732,410 |
| `6_plus_healthy_FINAL` | healthy-window contrast | 779,066 |

A rung's name is its checkpoint filename prefix (`<name>_seed<seed>.pt`), so
renaming one orphans its checkpoints.

## Features and weighting

**18 features**: 14 raw sensors + 4 geometry columns. The 14 causal
expanding-baseline columns were ablated against a no-baseline control and made
TEST_NASA worse on both subsets, most likely because they are redundant with
the healthy-window cross-attention, which already performs a learned
per-engine offset comparison. `baseline_feature_names` is emitted as an empty
list so loaders keep working.

**One weighting function for every subset.** The sample weight is
`1 + 3*(1 - RUL/125)^2`: exactly 1.0 at or above the cap, 4.0 at failure,
smooth throughout. It depends on capped RUL alone, so no dataset-specific
column can make two subsets train on different objectives. That is what
previously happened: FD001 emitted `onset`/`T_i`/`phase` and FD003 did not, and
the dataset silently substituted `onset=0, phase=1`, ramping FD003's weight
across the whole trajectory instead of only the degradation window.

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

Per dataset, under `$CMAPPS_OUT_ROOT/<dataset>/`:

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
