import numpy as np


def create_sequences(X, y, groups, window_size: int = 30, stride: int = 1):
    X_seq, y_seq = [], []

    for engine in np.unique(groups):
        idx = np.where(groups == engine)[0]
        X_engine = X[idx]
        y_engine = y[idx]

        if len(X_engine) != len(y_engine):
            raise ValueError("Sequence length mismatch for engine")

        for i in range(window_size - 1, len(X_engine), stride):
            start = i - window_size + 1
            window = X_engine[start : i + 1]

            if window.shape[0] < window_size:
                pad = np.zeros((window_size - window.shape[0], X.shape[1]))
                window = np.vstack((pad, window))

            X_seq.append(window)
            y_seq.append(y_engine[i])

    return np.array(X_seq), np.array(y_seq)


def sanity_check_sequences(X_seq, y_seq):
    if len(X_seq) != len(y_seq):
        raise ValueError("Mismatch in sequences and labels")
    if X_seq.ndim != 3:
        raise ValueError("X_seq must be 3D (samples, time, features)")

    print(f"Sequences OK: {X_seq.shape}")
