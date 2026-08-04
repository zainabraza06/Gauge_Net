"""
Shared configuration for the IGRFNet ablation study.

Every hyperparameter that the preprocessing, training and evaluation stages
have to agree on lives here, so a change cannot be applied to one stage and
silently missed by another (window length W and RUL_CAP in particular are
baked into both the cached geometry features and the trained checkpoints).

Paths default to directories inside the cloned repository, so the study runs
after a clone with no configuration. Override either one by environment
variable, or per run with --data-root / --out-root:

    CMAPPS_DATA_ROOT   raw C-MAPSS text files + pipeline_output/
                       default: <repo>/data
    CMAPPS_OUT_ROOT    checkpoints, geometry cache, result tables
                       default: <repo>/results/component_ablation_v2

Both default directories are git-ignored: the C-MAPSS archive is NASA's to
distribute, and checkpoints are far too large to version.
"""

import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTHONHASHSEED', '0')

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.environ.get('CMAPPS_DATA_ROOT', os.path.join(REPO_ROOT, 'data'))
# The final pipeline (no baseline features) and the unweighted loss differ from
# what everything under component_ablation/ was trained with, so results are
# written alongside rather than over them. The old checkpoints stay available
# but are NOT comparable: different loss, different feature count.
OUT_ROOT = os.environ.get('CMAPPS_OUT_ROOT',
                          os.path.join(REPO_ROOT, 'results', 'component_ablation_v2'))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── labels and sequence geometry ──
RUL_CAP = 125.0       # piecewise-linear RUL cap, standard for C-MAPSS
SCALE = 125.0         # network predicts RUL/SCALE, so targets sit in [0, 1]
W = 72                # causal window length fed to the encoder
H = 96                # hidden width
DROPOUT = 0.15

# ── optimisation ──
LR = 3e-4; LR_MIN = 1e-6; WD = 1e-3; GRAD_CLIP = 0.5
EPOCHS = 150; WARMUP_EP = 20; BATCH = 128
# Early stopping on validation NASA. 15 rather than 25: with six rungs x five
# seeds x two datasets the extra 10 epochs of waiting cost more than the runs
# they rescue, and selection is on the best epoch anyway, so a later recovery
# would have had to beat a checkpoint already saved.
PATIENCE = 15
VAL_FRAC = 0.20       # fraction of TRAIN engines held out per seed
END_TRIM_MAX = 5      # random end-of-life trim, augments run-to-failure data

# ── loss shaping ──
BETA = 1.5            # asymmetric Huber: penalty multiplier for late predictions
# No sample-weight constants: the loss is unweighted (see model.losses).

# ── geometry features ──
HEALTHY_FRAC = 0.20   # leading fraction of a trajectory treated as healthy

# ── validation snapshots and MC dropout ──
N_VAL_TRUNC = 8       # random truncation points drawn per validation engine
MC_PASSES_VAL = 10
MC_PASSES_TEST = 100

# ── test-time aggregation ──
USE_MULTIWINDOW = False
LAST_K = 9

# ── architecture ──
# Sensor-relationship branch: one token per sensor, holding that sensor's
# whole W-length window, over 2 layers of cross-sensor attention.
SENSOR_BRANCH_DIM = 32; SENSOR_BRANCH_HEADS = 4; SENSOR_BRANCH_LAYERS = 2
ATTN_HEADS = 2
NEAR_WINDOW = 8       # local attention span of the near path
# TCN reference backbone: 3 dilated causal conv layers (dilation 1, 2, 4),
# kernel 3 -> receptive field 15 cycles.
TCN_LAYERS = 3; TCN_KERNEL = 3

DATASETS = ['FD001', 'FD003']

# Preprocessed input produced by the preprocessing package, per subset.
# 'final' = no baseline features, no weighting: trajectory + capped RUL only.
# Must match preprocessing.pipeline.VERSION, which names the file written.
PIPELINE_VERSION = {'FD001': 'final', 'FD003': 'final'}

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
