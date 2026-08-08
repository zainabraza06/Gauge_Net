# IGRFNet — geometry-informed RUL prediction on C-MAPSS

Component-by-component ablation of a transformer-based Remaining Useful Life
(RUL) estimator on the NASA C-MAPSS turbofan datasets. The ladder is reported
on **FD001** and **FD003**; **FD002** and **FD004** preprocess as well, with
regime normalisation.

Every stage is causal: a prediction at cycle *t* is computed from cycles
≤ *t* of that engine alone, and every normalisation statistic is fitted on
the training split only.

---

## Layout

```
preprocessing/     raw C-MAPSS text  ->  leak-free feature pickle
  pipeline.py        z-score (global or per-regime), RUL cap, label
  __main__.py        python -m preprocessing FD001

model/             everything shared by training and evaluation
  config.py          hyperparameters and paths (env-overridable)
  metrics.py         NASA score, RMSE, MAE, mean error, MAPE, R², Accuracy(±k)
  losses.py          unweighted asymmetric Huber
  geometry.py        Riemannian / Wasserstein drift features (cached)
  windows.py         causal windowing, dataset, validation snapshots
  modules.py         attention blocks, sensor branch, fusion, pooling
  igrfnet.py         the ablatable network + build_model(cfg)
  data.py            load a subset, attach geometry, select a config's features
  inference.py       MC-dropout prediction, per-engine scoring, checkpoints

tests/
  smoke_test.py      end-to-end run on synthetic data, no NASA archive needed

ablation/          the study itself
  configs.py         the five-rung cumulative ladder + training run order
  train.py           python -m ablation.train
  ensembling.py      eight ways of combining a config's seeds
  stats.py           paired tests between adjacent rungs
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
and put the three files per subset you want in `data/`:

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

**A full run needs a GPU.** The ladder is 5 configs x 5 seeds x 4 datasets =
100 trainings, and `ablation.train` warns if no CUDA device is visible. To
sanity-check the pipeline on CPU first, run one rung with one seed:

```bash
python -m ablation.train --datasets FD001 --configs sensor_branch_linear_far --seeds 53
python -m ablation.evaluate --datasets FD001 --configs sensor_branch_linear_far
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

| Rung | Adds | Features | Params |
|---|---|---|---|
| `transformer` | single full-causal attention path (the reference) | 14 | 308,834 |
| `near_far` | local path + content-conditioned mixing gate (+ near→far cross-attention) | 14 | 570,656 |
| `geometry` | Riemannian / Wasserstein drift features | 18 | 571,040 |
| `sensor_branch` | iTransformer cross-sensor branch + cross-attention fusion | 18 | 745,984 |
| `sensor_branch_linear_far` | far path swapped to O(n) linear attention — **the settled model** | 18 | 745,984 |

The last step changes no parameter count at all: linear attention swaps four
projections for four projections. Whatever it does to the score, it does
without extra capacity, which is the cleanest kind of step this ladder has.

Five components were tried and are **not** rungs — a conv (TCN) baseline, an
FFT spectral gate, a reverse-time encoding, a healthy-window contrast, and
regime conditioning. `ablation/configs.py` records why each left; the short
version is that a rung has to be one component, has to move the metric beyond
seed noise, and has to be walkable on every subset.

A rung's name is its checkpoint filename prefix (`<name>_seed<seed>.pt`), so
renaming one orphans its checkpoints.

## Operating conditions

| Subset | Conditions | Normalisation |
|---|---|---|
| FD001, FD003 | one | a single z-score, fitted on train |
| FD002, FD004 | six | per-regime z-score: `KMeans(k=6)` over `op1/op2/op3`, fitted on train |

On FD002 and FD004 a change of flight condition moves a sensor far more than
degradation does, so one global z-score would leave regime changes looking
like wear. Each row is normalised against its own regime's train statistics
instead. Test rows are assigned to the nearest **train-fit** centre and
normalised with that regime's **train** statistics, so no test row reaches a
statistic.

This clustering is not the fault-mode discovery that was tried and dropped.
That clustered engines by degradation pattern, inferring a label the data does
not carry; this clusters rows by three measured columns with six known
settings. No fault-mode label is emitted or used anywhere — FD003 and FD004
have two modes each, and the model is expected to infer the mode from the
trajectory.

The regimes are handled entirely in preprocessing. An earlier version also
fed the regime id to the model as a learned embedding; it is gone, because a
rung FD001/FD003 cannot walk is not a rung, and the staircase is the same five
steps on all four subsets.

The pipeline declares which scheme a subset gets and then **cross-checks it
against the measured op-condition variance**, failing in either direction, so
a subset cannot quietly be normalised the wrong way. It also fails if any
regime has no train rows, or if test rows land in one — those rows would
otherwise be left at zero, reading as a perfectly average sensor.

`python -m ablation.train` still defaults to FD001 and FD003: every number
reported so far is on the single-condition pair. Pass `--datasets FD002` to
run the others.

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

## Reading the deltas

`ablation.evaluate` prints each rung against the one below it, paired by seed:
every rung trains on the same seed list, and a seed fixes the validation
split, the end-of-life trim and the initialisation, so a pair differs by the
component under test alone.

Four things are reported per step — paired t-test, Wilcoxon, Cohen's d, and a
bootstrap 95% CI on the mean difference — because no one of them is
trustworthy at five seeds. In particular **Wilcoxon cannot return p < 0.05
with five pairs at all** (its smallest attainable two-sided p is 0.0625), so a
non-significant Wilcoxon here is not evidence of no effect. Read the effect
size and whether the interval excludes zero.

Sign convention: `previous - current`, so a positive delta means the metric
fell when the component was added, and every metric in that table is
lower-is-better.

## Ensembling

The seeds of one config are combined seven ways: `mean`, `median`,
`trimmed_mean`, `val_weighted`, `softmax_sharp`, `softmax_soft`,
`rank_average`. A technique that cannot be computed from the members found is
skipped **with a recorded reason**, so a missing row in the results table is
never ambiguous. Combined predictions are written to
`<config>_ensemble_preds.npz`.

Each technique is then tested against the plain `mean`, **paired per engine**.
That is the one comparison in this study with real statistical power — the
pairing is over ~100 engines rather than 5 seeds — but it says which technique
scored better on this test set, not which generalises better: the
validation-weighted techniques have already seen the signal they are weighted
by.

Two things worth carrying into any write-up:

- **Validation-score weighting is only roughly comparable.** Each seed holds
  out different engines, so `best_val_nasa` carries split difficulty as well
  as model quality. It ranks members adequately; it is not a calibrated
  quality estimate.
- **Affine bias correction was removed.** `bias_corrected` fitted a
  correction from pooled validation residuals and applied it to the test
  predictions — a free parameter no other technique gets, fitted on the data
  the members were already selected on, so its margin could not be separated
  from that advantage.
- **`rank_average` ranks engines, not members.** Ranking across members is
  degenerate: every engine's rank column is a permutation of `0..n-1`, the
  mean rank is the same constant for every engine, and the result collapses to
  the median exactly. The implementation here ranks the engines each member
  scored, which keeps a member's ordering useful even if its RUL scale is off.
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
over-predicting RUL, the dangerous direction), `mape`, `r2` and
`Accuracy(±5/10/13/15/20)`.

`mape` floors its denominator at 1 cycle: true RUL reaches 0 at failure, so
the engines the score cares about most would otherwise each contribute an
unbounded term. Read it as a diagnostic, not a headline.

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
