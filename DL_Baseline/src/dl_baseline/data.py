import os
import pandas as pd


def load_cmapss_data(data_dir: str, dataset_name: str = "FD001"):
    columns = [
        "unit_nr",
        "time_cycles",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
    ]

    for i in range(1, 22):
        columns.append(f"sensor_{i}")

    train_path = os.path.join(data_dir, f"train_{dataset_name}.txt")
    test_path = os.path.join(data_dir, f"test_{dataset_name}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{dataset_name}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=columns)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=columns)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    train_df = train_df.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)
    test_df = test_df.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)

    if not train_df.groupby("unit_nr")["time_cycles"].is_monotonic_increasing.all():
        raise ValueError("Train data is not time-sorted per engine")

    return train_df, test_df, rul_df


def add_rul_column(df, max_rul: int = 125, clip: bool = True):
    df = df.copy()
    rul_values = []

    for engine in df["unit_nr"].unique():
        engine_df = df[df["unit_nr"] == engine]
        max_cycle = engine_df["time_cycles"].max()
        engine_rul = max_cycle - engine_df["time_cycles"].values

        if clip:
            engine_rul = engine_rul.clip(0, max_rul)

        rul_values.extend(engine_rul)

    df["RUL"] = rul_values

    if len(df["RUL"]) != len(df):
        raise ValueError("RUL column length mismatch")

    return df
