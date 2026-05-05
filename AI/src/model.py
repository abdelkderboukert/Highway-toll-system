"""
Model Management Module (Multi-Output Architecture)
Handles creation, initialization, and persistence of both LSTM and Random Forest models.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
import os
import boto3
import tempfile


# ==========================================
# 🧠 1. LSTM MODEL (For Forecasting)
# ==========================================
def build_lstm_model(time_steps, n_features, n_outputs):
    """Builds and compiles the LSTM Neural Network for multi-target forecasting."""
    logging.info(
        f"Initializing LSTM Model (Input shape: {time_steps} steps, {n_features} features, {n_outputs} outputs)"
    )
    model = Sequential(
        [
            LSTM(
                64,
                activation="relu",
                return_sequences=True,
                input_shape=(time_steps, n_features),
            ),
            Dropout(0.2),
            LSTM(32, activation="relu"),
            Dropout(0.2),
            Dense(n_outputs),  # <--- Automatically adjusts to predict 6 variables!
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def save_lstm_model(
    model,
    local_filepath="models/production_model.keras",
    s3_key="models/production_model.keras",
):
    """
    Saves the model locally if in development, or to S3 if in production.
    """
    # Check the environment (defaults to 'development' if not set)
    env = os.environ.get("ENVIRONMENT", "development").lower()

    if env == "production":
        # --- PRODUCTION: Push to S3 ---
        bucket_name = os.environ.get("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError(
                "S3_BUCKET_NAME environment variable is missing in production!"
            )

        s3_client = boto3.client("s3")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filepath = os.path.join(temp_dir, "temp_model.keras")
            model.save(temp_filepath)

            try:
                s3_client.upload_file(temp_filepath, bucket_name, s3_key)
                logging.info(
                    f"PRODUCTION: Model successfully pushed to s3://{bucket_name}/{s3_key}"
                )
            except Exception as e:
                logging.error(f"Failed to push model to S3: {e}")
                raise e
    else:
        # --- DEVELOPMENT: Save locally ---
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
        model.save(local_filepath)
        logging.info(f"DEVELOPMENT: Model saved locally to {local_filepath}")


def save_lstm_model_to_s3(model, bucket_name, s3_key="models/production_model.keras"):
    """
    Saves a Keras model to a temporary local file, uploads it to S3,
    and automatically cleans up the local file.
    """
    # Initialize the S3 client (AWS Batch automatically provides credentials via IAM roles)
    s3_client = boto3.client("s3")

    # Create a temporary directory that self-destructs when the block ends
    with tempfile.TemporaryDirectory() as temp_dir:
        local_filepath = os.path.join(temp_dir, "temp_model.keras")

        # 1. Save the model locally inside the temp folder
        model.save(local_filepath)
        logging.info("LSTM Model successfully compiled and saved to temporary storage.")

        # 2. Upload the file to S3
        try:
            s3_client.upload_file(local_filepath, bucket_name, s3_key)
            logging.info(
                f"LSTM Model successfully pushed to s3://{bucket_name}/{s3_key}"
            )
        except Exception as e:
            logging.error(f"Failed to push model to S3: {e}")
            raise e


# ==========================================
# 🌲 2. RANDOM FOREST MODEL (For Anomalies)
# ==========================================
# def build_rf_model(n_estimators=100, random_state=42):
#     """Initializes the Random Forest model (Native Multi-Output support)."""
#     logging.info(f"Initializing Random Forest model (Trees: {n_estimators})")
#     model = RandomForestRegressor(
#         n_estimators=n_estimators,
#         random_state=random_state,
#         n_jobs=-1
#     )
#     return model


def build_rf_model(n_estimators=50, random_state=42):
    logging.info(
        f"Initializing Random Forest model (Trees: {n_estimators}) with RAM limits"
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,  # Reduced from 100 to 50
        max_depth=15,  # Stops trees from growing infinitely and eating RAM
        min_samples_split=10,  # Prevents calculating splits on tiny data fragments
        random_state=random_state,
        n_jobs=2,  # Changed from -1. Restricts to 2 CPU cores to prevent memory duplication
    )
    return model


def save_rf_model(model, filepath="models/rf_model.joblib"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Random Forest Model successfully saved to {filepath}")


def save_rf_model(
    model, local_filepath="models/rf_model.joblib", s3_key="models/rf_model.joblib"
):
    """
    Saves the model locally if in development, or to S3 if in production.
    """
    # Check the environment (defaults to 'development' if not set)
    env = os.environ.get("ENVIRONMENT", "development").lower()

    if env == "production":
        # --- PRODUCTION: Push to S3 ---
        bucket_name = os.environ.get("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError(
                "S3_BUCKET_NAME environment variable is missing in production!"
            )

        s3_client = boto3.client("s3")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filepath = os.path.join(temp_dir, "temp_model.joblib")
            joblib.dump(model, temp_filepath)

            try:
                s3_client.upload_file(temp_filepath, bucket_name, s3_key)
                logging.info(
                    f"PRODUCTION: Model successfully pushed to s3://{bucket_name}/{s3_key}"
                )
            except Exception as e:
                logging.error(f"Failed to push model to S3: {e}")
                raise e
    else:
        # --- DEVELOPMENT: Save locally ---
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
        joblib.dump(model, local_filepath)
        logging.info(f"DEVELOPMENT: Model saved locally to {local_filepath}")
