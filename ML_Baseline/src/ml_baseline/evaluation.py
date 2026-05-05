import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .metrics import phm08_competition_score, phm08_score


def evaluate_on_test_set(train_pipeline, test_df, rul_df, engine_ids, feature_cols):
    X_test = test_df[feature_cols].values
    groups = test_df["unit_nr"].values
    time_cycles = test_df["time_cycles"].values

    y_pred_full = train_pipeline.predict(X_test, groups=groups, time_cycles=time_cycles)

    true_final = []
    pred_final = []

    rul_values = rul_df["rul"].values

    for i, engine in enumerate(engine_ids):
        mask = groups == engine
        idx = np.where(mask)[0]
        last_idx = idx[np.argmax(time_cycles[mask])]

        true_final.append(rul_values[i])
        pred_final.append(y_pred_full[last_idx])

    true_final = np.array(true_final)
    pred_final = np.array(pred_final)

    rmse = np.sqrt(mean_squared_error(true_final, pred_final))
    mae = mean_absolute_error(true_final, pred_final)
    phm_total = phm08_competition_score(true_final, pred_final)

    engine_scores = [phm08_score(t, p) for t, p in zip(true_final, pred_final)]

    print("\n====================================")
    print("FINAL TEST RESULTS (PHM08)")
    print("====================================")

    print(f"PHM08 Total Score : {phm_total:.3f}")
    print(f"RMSE              : {rmse:.3f}")
    print(f"MAE               : {mae:.3f}")
    print(f"Avg Engine Score  : {np.mean(engine_scores):.3f}")
    print(f"Best Engine Score : {np.min(engine_scores):.3f}")
    print(f"Worst Engine Score: {np.max(engine_scores):.3f}")

    return {
        "phm08_score": float(phm_total),
        "rmse": float(rmse),
        "mae": float(mae),
        "engine_scores": [float(x) for x in engine_scores],
        "y_true": true_final,
        "y_pred": pred_final,
    }
