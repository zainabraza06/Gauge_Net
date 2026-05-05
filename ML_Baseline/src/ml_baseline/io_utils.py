import json
import os
from datetime import datetime
import joblib


def save_experiment_results(results, pipeline, dataset_name, results_dir, models_dir, seed):
    model_path = os.path.join(models_dir, f"{dataset_name}_model.pkl")
    results_path = os.path.join(results_dir, f"{dataset_name}_results.json")

    joblib.dump(pipeline, model_path)

    clean_results = {
        "phm08_score": float(results["phm08_score"]),
        "rmse": float(results["rmse"]),
        "mae": float(results["mae"]),
        "engine_scores": [float(x) for x in results["engine_scores"]],
        "timestamp": str(datetime.now()),
        "seed": seed,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=4)

    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
