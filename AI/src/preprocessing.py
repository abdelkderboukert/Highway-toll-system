"""
Data Preprocessing Module

Performs comprehensive data cleaning and transformation including handling missing
values, date formatting, removal of invalid data points, and handling time-series gaps.
"""

import pandas as pd
import numpy as np
import logging


def clean_data(df):
    """
    Clean and preprocess oil well production dataset.

    Processing steps:
    1. Remove irrelevant columns
    2. Convert date strings to datetime objects
    3. Replace negative values with NaN (physically invalid)
    4. Resample dates to continuous daily frequency (handling gaps)
    5. Impute missing values using forward-fill and back-fill by well
    6. Fill remaining NaN values with 0

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing oil well production data.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for modeling.
    """
    logging.info("Starting data preprocessing...")

    # Remove irrelevant columns
    df = df.drop(columns=["prodindex"], errors="ignore")

    # Convert DATE column to datetime format
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Replace physically impossible negative values with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df.loc[df[col] < 0, col] = np.nan

    # --- UPGRADE: HANDLE MISSING DATE GAPS (RESAMPLING) ---
    logging.info("Fixing date gaps: Resampling to continuous daily frequency...")
    df = df.set_index("DATE")
    resampled_list = []

    # We do this well by well so we don't accidentally create dates between the end
    # of one well and the start of another!
    for well, group in df.groupby("WELL"):
        # .resample('1D') forces the timeline to be perfectly continuous day-by-day.
        # .asfreq() injects NaN (empty) rows for any missing calendar days.
        group_resampled = group.resample("1D").asfreq()
        group_resampled["WELL"] = (
            well  # Ensure the empty rows still know which well they belong to
        )
        resampled_list.append(group_resampled)

    # Combine all wells back together and turn the DATE index back into a column
    df = pd.concat(resampled_list).reset_index()
    # ------------------------------------------------------

    # Impute missing values (This will now fill the newly created empty date rows!)
    logging.info("Filling missing values by well...")
    cols_to_fill = df.columns.drop(["WELL", "DATE"])

    # Forward fill (copy yesterday's data to today), then backward fill if needed
    df[cols_to_fill] = df.groupby("WELL")[cols_to_fill].ffill().bfill()

    # Fill any remaining NaN values with 0
    df_clean = df.fillna(0)

    # Sort data chronologically by well and date just to be absolutely certain
    df_clean = df_clean.sort_values(by=["WELL", "DATE"]).reset_index(drop=True)

    logging.info("Preprocessing complete.")
    return df_clean
