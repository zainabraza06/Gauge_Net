import argparse
import json
import os
import warnings

from .config import Config, FEATURE_COLS, resolve_paths
from .data import add_rul_column, load_cmapss_data
from .evaluation import evaluate_on_test_set
from .io_utils import save_experiment_results
from .pipeline import cross_validate_groupkfold, train_final_model
from .seed import set_global_seed


def run_full_experiment(dataset_name, model_type, config, paths, use_temporal_features):
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {dataset_name} | MODEL: {model_type}")
    print("=" * 80)

    train_df, test_df, rul_df, engine_ids = load_cmapss_data(
        paths["data_dir"], dataset_name
    )

    train_df = add_rul_column(
        train_df, max_rul=config.max_rul, clip=config.clip_rul
    )

    X = train_df[FEATURE_COLS].values
    y = train_df["RUL"].values
    groups = train_df["unit_nr"].values
    time_cycles = train_df["time_cycles"].values

    cv_results = cross_validate_groupkfold(
        X,
        y,
        groups,
        time_cycles,
        model_type=model_type,
        n_splits=config.cv_folds,
        n_features=config.n_features,
        seed=config.seed,
        use_temporal_features=use_temporal_features,
        rolling_windows=config.rolling_windows,
        n_lags=config.n_lags,
        use_rolling_std=config.use_rolling_std,
    )

    pipeline = train_final_model(
        X,
        y,
        groups,
        time_cycles,
        model_type=model_type,
        n_features=config.n_features,
        seed=config.seed,
        use_temporal_features=use_temporal_features,
        rolling_windows=config.rolling_windows,
        n_lags=config.n_lags,
        use_rolling_std=config.use_rolling_std,
    )

    results = evaluate_on_test_set(
        train_pipeline=pipeline,
        test_df=test_df,
        rul_df=rul_df,
        engine_ids=engine_ids,
        feature_cols=FEATURE_COLS,
    )

    save_experiment_results(
        results,
        pipeline,
        dataset_name,
        results_dir=paths["results_dir"],
        models_dir=paths["models_dir"],
        seed=config.seed,
    )

    return {
        "dataset": dataset_name,
        "model": model_type,
        "cv": cv_results,
        "test": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMAPSS ML baseline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to CMAPSS data directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run (e.g., FD001 FD002)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "rf", "xgboost"],
        help="Models to run",
    )
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-summary", action="store_true")
    parser.add_argument(
        "--use-temporal-features",
        action="store_true",
        help="Enable preset temporal feature engineering",
    )

    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config = Config()

    if args.cv_folds is not None:
        config.cv_folds = args.cv_folds
    if args.n_features is not None:
        config.n_features = args.n_features
    if args.seed is not None:
        config.seed = args.seed
    if args.datasets is not None:
        config.datasets = args.datasets

    set_global_seed(config.seed)
    paths = resolve_paths(project_root, data_dir=args.data_dir)

    all_results = {}

    for ds in config.datasets:
        all_results[ds] = {}
        for model_name in args.models:
            res = run_full_experiment(
                ds,
                model_type=model_name,
                config=config,
                paths=paths,
                use_temporal_features=args.use_temporal_features,
            )
            all_results[ds][model_name] = res

    if args.save_summary:
        summary_path = os.path.join(paths["results_dir"], "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
