import numpy as np


def phm08_score(y_true, y_pred):
    d = y_pred - y_true

    if d < 0:
        return np.exp(-d / 13.0) - 1.0

    return np.exp(d / 10.0) - 1.0


def phm08_competition_score(y_true, y_pred):
    total = 0.0

    for t, p in zip(y_true, y_pred):
        d = p - t

        if d < 0:
            total += np.exp(-d / 13.0) - 1.0
        else:
            total += np.exp(d / 10.0) - 1.0

    return total
