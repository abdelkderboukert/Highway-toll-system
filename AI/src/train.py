"""
Model Training and Evaluation Module (Multi-Output)
Handles scaling, 3D sequence generation, training, and evaluation for multiple targets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import logging
import os
from src.model import build_lstm_model, build_rf_model


def create_sequences(df, features, targets, time_steps):
    """Creates 3D time windows safely grouped by WELL for multiple targets."""
    X, y = [], []
    for well_name, group in df.groupby("WELL"):
        group_features = group[features].values
        group_targets = group[targets].values
        for i in range(len(group) - time_steps):
            X.append(group_features[i : i + time_steps])
            y.append(group_targets[i + time_steps])
    return np.array(X), np.array(y)


# ==========================================
# 🧠 1. LSTM TRAINING LOGIC
# ==========================================
def train_and_evaluate_lstm(df, features, targets, time_steps, test_size, random_state):
    logging.info("--- STARTING LSTM (FORECASTER) PIPELINE ---")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df_scaled = df.copy()
    df_scaled[features] = feature_scaler.fit_transform(df[features])
    df_scaled[targets] = target_scaler.fit_transform(df[targets])

    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_scaler, "models/feature_scaler.joblib")
    joblib.dump(target_scaler, "models/target_scaler.joblib")

    X, y = create_sequences(df_scaled, features, targets, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    model = build_lstm_model(
        time_steps=time_steps, n_features=len(features), n_outputs=len(targets)
    )
    model.fit(
        X_train, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=1
    )

    predictions_scaled = model.predict(X_test)

    # Notice: No reshape or flatten! Works for 6 columns perfectly.
    y_test_real = target_scaler.inverse_transform(y_test)
    predictions_real = target_scaler.inverse_transform(predictions_scaled)

    metrics = {
        "Model": f"LSTM ({len(targets)} Outputs)",
        "R2_Score": round(r2_score(y_test_real, predictions_real), 4),
        "MAE": round(mean_absolute_error(y_test_real, predictions_real), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test_real, predictions_real)), 4),
    }
    return model, y_test_real, predictions_real, metrics


# ==========================================
# 🌲 2. RANDOM FOREST TRAINING LOGIC
# ==========================================
def train_and_evaluate_rf(df, features, targets, test_size, random_state):
    logging.info("--- STARTING RANDOM FOREST PIPELINE ---")

    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    model = build_rf_model(random_state=random_state)
    logging.info("Training Multi-Output Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Model": f"Random Forest ({len(targets)} Outputs)",
        "R2_Score": round(r2_score(y_test, predictions), 4),
        "MAE": round(mean_absolute_error(y_test, predictions), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, predictions)), 4),
    }
    return model, y_test.values, predictions, metrics
