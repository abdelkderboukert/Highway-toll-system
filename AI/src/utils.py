"""
Utility Module
Provides helper functions for configuration and feature leakage checking for Multi-Output models.
"""

import yaml
import logging
import pandas as pd
import requests
import os
import logging


def notify_django_to_reload():
    """Pings the Django backend to tell it to fetch the new S3 models."""

    # In production, these come from your AWS Batch Environment Variables
    webhook_url = os.environ.get(
        "DJANGO_WEBHOOK_URL", "http://127.0.0.1:8000/api/webhooks/reload-models/"
    )
    webhook_secret = os.environ.get("WEBHOOK_SECRET", "my-super-secret-dev-token")

    logging.info(f"Notifying Django backend at {webhook_url}...")

    headers = {"X-Webhook-Token": webhook_secret}

    try:
        response = requests.post(webhook_url, headers=headers)

        if response.status_code == 200:
            logging.info("SUCCESS: Django acknowledged and reloaded the model in RAM!")
        else:
            logging.error(
                f"Django returned an error: {response.status_code} - {response.text}"
            )

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to Django webhook: {e}")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path="config/config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logging.info("Configuration loaded successfully.")
            return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise


def check_feature_leakage(df):
    config = load_config()
    safe_list = []

    # Handle both single string and lists
    targets = config["pipeline"]["target_col"]
    if isinstance(targets, str):
        targets = [targets]

    features = config["pipeline"]["feature_cols"]

    print(f"\n{'FEATURE':<15} | {'MAX CORRELATION (ALL TARGETS)':<30} | {'STATUS'}")
    print("-" * 80)

    for col in features:
        if col in df.columns:
            # Check correlation against ALL target variables and pick the highest one
            max_corr = 0
            for target in targets:
                corr = abs(df[col].corr(df[target]))
                if pd.notna(corr) and corr > max_corr:
                    max_corr = corr

            if max_corr > 0.95:
                status = "❌ HIGH LEAKAGE RISK"
            elif max_corr > 0.80:
                status = "STRONG DRIVER"
            else:
                status = "OK"
                safe_list.append(col)

            print(f"{col:<15} | {max_corr:>30.4f} | {status}")
        else:
            print(f"{col:<15} | {'COLUMN MISSING':>30} | WARN")

    config["pipeline"]["safe_features"] = safe_list

    try:
        with open("config/config.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        logging.info(
            f"Success: config.yaml updated with {len(safe_list)} safe features."
        )
    except Exception as e:
        logging.error(f"Could not update config.yaml: {e}")
