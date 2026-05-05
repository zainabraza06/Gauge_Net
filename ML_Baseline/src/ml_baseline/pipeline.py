import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from .features import FeatureScaler, FeatureSelector, TemporalFeatureEngineer
from .metrics import phm08_score
from .models import build_model


class SimplePipeline:
    def __init__(self, scaler, selector, model, temporal_engineer=None):
        self.scaler = scaler
        self.selector = selector
        self.model = model
        self.temporal_engineer = temporal_engineer

    def predict(self, X, groups=None, time_cycles=None):
        if self.temporal_engineer is not None:
            if groups is None or time_cycles is None:
                raise ValueError("groups and time_cycles are required for temporal features")
            X = self.temporal_engineer.transform(X, groups=groups, time_cycles=time_cycles)
        X_scaled = self.scaler.transform(X)
        X_selected = self.selector.transform(X_scaled)
        return self.model.predict(X_selected)


def cross_validate_groupkfold(
    X,
    y,
    groups,
    time_cycles,
    model_type: str,
    n_splits: int,
    n_features: int,
    seed: int,
    use_temporal_features: bool,
    rolling_windows,
    n_lags: int,
    use_rolling_std: bool,
):
    gkf = GroupKFold(n_splits=n_splits)
    rmse_list = []
    phm_list = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        g_train = groups[train_idx]
        g_val = groups[val_idx]
        t_train = time_cycles[train_idx]
        t_val = time_cycles[val_idx]

        if use_temporal_features:
            temporal_engineer = TemporalFeatureEngineer(
                window_sizes=rolling_windows,
                n_lags=n_lags,
                use_rolling_std=use_rolling_std,
            )
            temporal_engineer.fit(X_train)
            X_train = temporal_engineer.transform(
                X_train, groups=g_train, time_cycles=t_train
            )
            X_val = temporal_engineer.transform(X_val, groups=g_val, time_cycles=t_val)

        scaler = FeatureScaler()
        X_train = scaler.fit(X_train).transform(X_train)
        X_val = scaler.transform(X_val)

        selector = FeatureSelector(n_features=n_features)
        X_train = selector.fit(X_train, y_train).transform(X_train)
        X_val = selector.transform(X_val)

        model = build_model(model_type, seed=seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        unique_engines = np.unique(g_val)
        phm_score = 0.0

        for eng in unique_engines:
            mask = g_val == eng
            idx = np.where(mask)[0]
            last = idx[np.argmax(t_val[mask])]
            phm_score += phm08_score(y_val[last], y_pred[last])

        phm_score /= len(unique_engines)

        rmse_list.append(rmse)
        phm_list.append(phm_score)

        print(f"Fold {fold}: RMSE={rmse:.3f}, PHM={phm_score:.3f}")

    return {
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "phm_mean": float(np.mean(phm_list)),
        "phm_std": float(np.std(phm_list)),
        "fold_phm": [float(x) for x in phm_list],
    }


def train_final_model(
    X,
    y,
    groups,
    time_cycles,
    model_type: str,
    n_features: int,
    seed: int,
    use_temporal_features: bool,
    rolling_windows,
    n_lags: int,
    use_rolling_std: bool,
):
    temporal_engineer = None

    if use_temporal_features:
        temporal_engineer = TemporalFeatureEngineer(
            window_sizes=rolling_windows,
            n_lags=n_lags,
            use_rolling_std=use_rolling_std,
        )
        temporal_engineer.fit(X)
        X = temporal_engineer.transform(X, groups=groups, time_cycles=time_cycles)

    scaler = FeatureScaler()
    X_scaled = scaler.fit(X).transform(X)

    selector = FeatureSelector(n_features=n_features)
    X_selected = selector.fit(X_scaled, y).transform(X_scaled)

    model = build_model(model_type, seed=seed)
    model.fit(X_selected, y)

    return SimplePipeline(
        scaler=scaler,
        selector=selector,
        model=model,
        temporal_engineer=temporal_engineer,
    )
