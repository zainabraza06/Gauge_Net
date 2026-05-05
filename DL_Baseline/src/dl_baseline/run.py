import argparse
import json
import os

import torch

from .config import Config, resolve_paths
from .data import add_rul_column, load_cmapss_data
from .dataset import create_loaders, split_by_engine
from .evaluation import evaluate_on_test, predict
from .features import fit_scaler, get_feature_columns, transform_data
from .metrics import calculate_metrics
from .models import CNN1D, LSTMModel, MLP
from .sequences import create_sequences
from .seed import set_global_seed
from .train import train_model


def run_dl_experiments(dataset_name, config, paths, device):
    print("\n" + "=" * 80)
    print(f"DEEP LEARNING EXPERIMENTS: {dataset_name}")
    print("=" * 80)

    train_df, test_df, rul_df = load_cmapss_data(paths["data_dir"], dataset_name)
    train_df = add_rul_column(train_df, max_rul=config.max_rul, clip=True)

    feature_cols = get_feature_columns(train_df)

    scaler = fit_scaler(train_df, feature_cols)
    X_train_scaled = transform_data(train_df, feature_cols, scaler)
    y_train = train_df["RUL"].values
    groups_train = train_df["unit_nr"].values

    train_mask, val_mask = split_by_engine(
        groups_train, val_ratio=config.val_ratio, seed=config.seed
    )

    X_train_original = X_train_scaled[train_mask]
    y_train_original = y_train[train_mask]
    groups_train_original = groups_train[train_mask]

    X_val_original = X_train_scaled[val_mask]
    y_val_original = y_train[val_mask]
    groups_val_original = groups_train[val_mask]

    X_train_seq, y_train_seq = create_sequences(
        X_train_original,
        y_train_original,
        groups_train_original,
        window_size=config.window_size,
    )
    X_val_seq, y_val_seq = create_sequences(
        X_val_original,
        y_val_original,
        groups_val_original,
        window_size=config.window_size,
    )

    train_loader, val_loader = create_loaders(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, batch_size=config.batch_size
    )

    val_metrics = {}
    test_metrics = {}

    models_to_run = {"MLP": MLP, "CNN": CNN1D, "LSTM": LSTMModel}

    for model_name, ModelClass in models_to_run.items():
        print(f"\n--- Training {model_name} ---")

        if model_name == "MLP":
            model = ModelClass(X_train_seq.shape[1] * X_train_seq.shape[2])
        elif model_name == "CNN":
            model = ModelClass(X_train_seq.shape[2])
        elif model_name == "LSTM":
            model = ModelClass(X_train_seq.shape[2])
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        model = train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=config.epochs,
            lr=config.learning_rate,
            patience=config.patience,
        )

        val_preds = predict(model, X_val_seq, device=device)
        val_metrics[model_name] = calculate_metrics(y_val_seq, val_preds)
        print(
            "Validation "
            f"{model_name}: RMSE={val_metrics[model_name]['RMSE']:.2f}, "
            f"NASA={val_metrics[model_name]['NASA_Score']:.2f}"
        )

        test_result, _ = evaluate_on_test(
            model,
            test_df,
            rul_df,
            scaler,
            feature_cols,
            device=device,
            max_rul=config.max_rul,
            window_size=config.window_size,
        )
        test_metrics[model_name] = test_result
        print(
            "Test "
            f"{model_name}: RMSE={test_metrics[model_name]['RMSE']:.2f}, "
            f"NASA={test_metrics[model_name]['NASA_Score']:.2f}"
        )

    results_path = os.path.join(
        paths["results_dir"], f"{dataset_name}_dl_experiment_results.json"
    )
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=4)
    print(f"\nDeep Learning results saved to: {results_path}")

    return val_metrics, test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMAPSS DL baseline")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config = Config()

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.window_size is not None:
        config.window_size = args.window_size
    if args.seed is not None:
        config.seed = args.seed
    if args.datasets is not None:
        config.datasets = args.datasets

    set_global_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Seed: {config.seed}")
    print(f"Device: {device}")
    print(f"Max RUL clip: {config.max_rul}")

    paths = resolve_paths(project_root, data_dir=args.data_dir)

    all_results = {}

    for ds in config.datasets:
        print("\n" + "=" * 80)
        print(f"RUNNING DATASET: {ds}")
        print("=" * 80)

        try:
            _, test_metrics = run_dl_experiments(ds, config=config, paths=paths, device=device)
            all_results[ds] = test_metrics
        except Exception as exc:
            print(f"Error in {ds}: {exc}")
            continue

    summary_path = os.path.join(paths["results_dir"], "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
