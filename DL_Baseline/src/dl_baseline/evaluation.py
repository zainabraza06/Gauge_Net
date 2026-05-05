import numpy as np
import torch

from .metrics import calculate_metrics


def predict(model, X, device, batch_size: int = 256):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            p = model(xb).cpu().numpy()
            preds.extend(p)

    return np.array(preds)


def evaluate_on_test(
    model,
    test_df,
    rul_df,
    scaler,
    feature_cols,
    device,
    max_rul: int,
    window_size: int = 30,
):
    X_test = test_df[feature_cols].values
    X_test = scaler.transform(X_test)

    X_seq = []
    engines = np.unique(test_df["unit_nr"])

    for eng in engines:
        idx = np.where(test_df["unit_nr"] == eng)[0]
        idx = idx[np.argsort(test_df.loc[idx, "time_cycles"].values)]
        X_eng = X_test[idx]

        if len(X_eng) >= window_size:
            window = X_eng[-window_size:]
        else:
            pad = np.zeros((window_size - len(X_eng), X_test.shape[1]))
            window = np.vstack((pad, X_eng))

        X_seq.append(window)

    X_seq = np.array(X_seq)
    preds = predict(model, X_seq, device=device)
    preds = np.clip(preds, 0, max_rul)

    y_true = rul_df["RUL"].values
    metrics = calculate_metrics(y_true, preds)

    return metrics, preds
