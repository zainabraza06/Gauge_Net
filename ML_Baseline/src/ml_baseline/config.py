import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    seed: int = 42
    max_rul: int = 125
    clip_rul: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10])
    n_lags: int = 1
    use_rolling_std: bool = False
    n_features: int = 15
    cv_folds: int = 5
    selector_cv_folds: int = 3
    datasets: List[str] = field(default_factory=lambda: ["FD002", "FD004"])


FEATURE_COLS = (
    ["op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def resolve_paths(project_root: str, data_dir: str | None = None):
    base_dir = os.path.abspath(project_root)
    data_dir = data_dir or os.path.join(base_dir, "cmapps")
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "models")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "results_dir": results_dir,
        "models_dir": models_dir,
    }
