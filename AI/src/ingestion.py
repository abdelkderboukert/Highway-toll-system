# """
# Data Ingestion Module

# Handles loading and initial import of raw data from CSV files for subsequent
# processing and analysis.
# """

# import pandas as pd
# import logging


# def load_raw_data(filepath):
#     """
#     Load raw CSV data into a Pandas DataFrame.

#     Parameters
#     ----------
#     filepath : str
#         Path to the CSV file containing raw data.

#     Returns
#     -------
#     pd.DataFrame
#         Loaded dataframe with raw data.

#     Raises
#     ------
#     FileNotFoundError
#         If the specified file does not exist.
#     """
#     logging.info(f"Loading raw data from {filepath}...")
#     df = pd.read_csv(filepath)
#     logging.info(f"Data loaded successfully. Shape: {df.shape}")
#     return df

"""
Data Ingestion Module

Handles loading and initial import of raw data from CSV files for subsequent
processing and analysis.
"""

import os
import pandas as pd
import logging
import boto3
import tempfile


def load_raw_data(
    local_filepath="data/raw_data.csv", s3_key="exports/training_data.csv"
):
    """
    Load raw CSV data into a Pandas DataFrame.
    Reads locally in development or from S3 in production.

    Parameters
    ----------
    local_filepath : str
        Path to the local CSV file (used in development).
    s3_key : str
        The S3 key/path to the CSV file (used in production).

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with raw data.
    """
    env = os.environ.get("ENVIRONMENT", "development").lower()

    if env == "production":
        # --- PRODUCTION: Download from S3 ---
        bucket_name = os.environ.get("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError(
                "S3_BUCKET_NAME environment variable is missing in production!"
            )

        logging.info(
            f"PRODUCTION: Downloading raw data from s3://{bucket_name}/{s3_key}..."
        )
        s3_client = boto3.client("s3")

        # Create a temporary file to hold the CSV just long enough for Pandas to read it
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as temp_file:
            try:
                s3_client.download_file(bucket_name, s3_key, temp_file.name)
                df = pd.read_csv(temp_file.name)
                logging.info(f"SUCCESS: Data loaded from S3. Shape: {df.shape}")
                return df
            except Exception as e:
                logging.error(f"Failed to download or read data from S3: {e}")
                raise e
    else:
        # --- DEVELOPMENT: Load from Local Filesystem ---
        logging.info(f"DEVELOPMENT: Loading raw data locally from {local_filepath}...")
        try:
            df = pd.read_csv(local_filepath)
            logging.info(f"SUCCESS: Data loaded locally. Shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            logging.error(f"Local file not found at {local_filepath}")
            raise e
