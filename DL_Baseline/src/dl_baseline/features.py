from sklearn.preprocessing import StandardScaler


def get_feature_columns(df):
    op_cols = [c for c in df.columns if c.startswith("op_setting")]
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    selected_sensors = []
    removed_sensors = []

    for col in sensor_cols:
        std = df[col].std()
        if std < 1e-4:
            removed_sensors.append(col)
        else:
            selected_sensors.append(col)

    if removed_sensors:
        print(f"Removed {len(removed_sensors)} low-variance sensors")

    print(f"Operational features: {len(op_cols)}")
    print(f"Sensor features     : {len(selected_sensors)}")

    return op_cols + selected_sensors


def fit_scaler(train_df, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def transform_data(df, feature_cols, scaler):
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    return X_scaled
