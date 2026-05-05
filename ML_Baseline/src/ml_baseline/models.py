import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def build_model(model_type: str, seed: int):
    if model_type == "ridge":
        return Ridge(alpha=10.0, random_state=seed)

    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        )

    if model_type == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
        )

    raise ValueError("Unknown model type")
