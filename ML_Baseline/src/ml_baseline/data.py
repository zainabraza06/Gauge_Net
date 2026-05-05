import os
import numpy as np
import pandas as pd


def load_cmapss_data(data_dir: str, subset: str):
    """
    Load CMAPSS FD00X dataset with checks and explicit columns.
    """

    col_names = (
        ["unit_nr", "time_cycles"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")

    for path in [train_path, test_path, rul_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])

    train_df = train_df.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)
    test_df = test_df.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)

    if (train_df.groupby("unit_nr")["time_cycles"].diff().dropna() < 0).any():
        raise ValueError("Train data is not time-sorted per engine")

    if (test_df.groupby("unit_nr")["time_cycles"].diff().dropna() < 0).any():
        raise ValueError("Test data is not time-sorted per engine")

    test_engine_ids = np.sort(test_df["unit_nr"].unique())

    if len(rul_df) != len(test_engine_ids):
        raise ValueError(
            f"RUL mismatch: {len(rul_df)} RUL values vs {len(test_engine_ids)} engines"
        )

    return train_df, test_df, rul_df, test_engine_ids


def add_rul_column(train_df, max_rul: int = 125, clip: bool = True):
    df = train_df.copy()
    max_cycle = df.groupby("unit_nr")["time_cycles"].transform("max")
    df["RUL"] = max_cycle - df["time_cycles"]

    if clip:
        df["RUL"] = df["RUL"].clip(0, max_rul)

    return df


def prepare_test_data(test_df):
    return test_df.copy()
