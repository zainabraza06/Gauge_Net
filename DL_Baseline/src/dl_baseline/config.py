import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    seed: int = 42
    max_rul: int = 125
    window_size: int = 30
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    patience: int = 5
    val_ratio: float = 0.2
    datasets: List[str] = field(default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"])


def resolve_paths(project_root: str, data_dir: str | None = None):
    base_dir = os.path.abspath(project_root)
    data_dir = data_dir or os.path.join(base_dir, "cmapps")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "results_dir": results_dir,
    }
