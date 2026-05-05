import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Causal feature engineering for CMAPSS.
    Builds original, rolling mean, optional rolling std, and lag features.
    """

    def __init__(self, window_sizes=None, n_lags: int = 1, use_rolling_std: bool = False):
        self.window_sizes = window_sizes or [5, 10]
        self.n_lags = n_lags
        self.use_rolling_std = use_rolling_std

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X, groups, time_cycles):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        outputs = np.zeros((n_samples, self._output_dim(n_features)))

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            order = np.argsort(time_cycles[idx])
            idx = idx[order]

            data = X[idx]
            df = pd.DataFrame(data).ffill().fillna(0.0).values

            engine_out = []
            engine_out.append(df)

            for w in self.window_sizes:
                roll_mean = (
                    pd.DataFrame(df)
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .values
                )
                engine_out.append(roll_mean)

                if self.use_rolling_std:
                    roll_std = (
                        pd.DataFrame(df)
                        .rolling(window=w, min_periods=1)
                        .std(ddof=0)
                        .fillna(0.0)
                        .values
                    )
                    engine_out.append(roll_std)

            for lag in range(1, self.n_lags + 1):
                lagged = np.zeros_like(df)
                lagged[lag:] = df[:-lag]
                engine_out.append(lagged)

            outputs[idx] = np.hstack(engine_out)

        return outputs

    def _output_dim(self, n_features: int):
        base = 1 + len(self.window_sizes)
        if self.use_rolling_std:
            base += len(self.window_sizes)
        return n_features * (base + self.n_lags)


class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


class FeatureSelector:
    def __init__(self, n_features: int = 15):
        self.n_features = n_features
        self.selected_idx_ = None

    def fit(self, X, y):
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        importance = np.abs(model.coef_)
        self.selected_idx_ = np.argsort(importance)[-self.n_features:]
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]
