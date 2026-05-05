"""
Visualization Module

Generates publication-quality plots for model prediction analysis and
feature importance visualization.
"""

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for server environments
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import os

import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend for server environments

# Create output directory for visualizations
os.makedirs("plots", exist_ok=True)


def plot_actual_vs_predicted(y_true, y_pred, filepath="plots/actual_vs_predicted.png"):
    """
    Generate scatter plot comparing actual vs predicted production values.

    Includes a diagonal reference line indicating perfect predictions.

    Parameters
    ----------
    y_true : array-like
        Actual production values from test set.
    y_pred : array-like
        Model predictions for test set.
    filepath : str, optional
        Output path for saved plot (default: "plots/actual_vs_predicted.png").

    Returns
    -------
    None
        Saves plot to file.
    """
    logging.info("Generating Actual vs Predicted plot...")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color="blue")

    # Add perfect prediction reference line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

    plt.xlabel("Actual Production (W_GAS)")
    plt.ylabel("Predicted Production (W_GAS)")
    plt.title("Actual vs Predicted Production")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Plot saved to {filepath}")


def plot_feature_importance(model, features, filepath="plots/feature_importance.png"):
    """
    Generate horizontal bar plot of feature importance scores.

    Visualizes which operational parameters have the greatest influence on
    production predictions based on Random Forest model.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model with feature importance attributes.
    features : list
        List of feature names corresponding to model input.
    filepath : str, optional
        Output path for saved plot (default: "plots/feature_importance.png").

    Returns
    -------
    None
        Saves plot to file.
    """
    logging.info("Generating Feature Importance plot...")
    importances = model.feature_importances_

    # Sort features by importance in descending order
    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=sorted_importances,
        y=sorted_features,
        hue=sorted_features,
        legend=False,
        palette="viridis",
    )
    plt.title("Feature Importance - Key Production Drivers")
    plt.xlabel("Relative Importance")
    plt.ylabel("Operational Parameters")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Plot saved to {filepath}")


def plot_well_time_series(df, well_name, target_col="W_GAS", filepath=None):
    """
    Generate time series plot of production profile for a specific well.

    Visualizes production trends over time for individual well analysis
    and temporal pattern identification.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset with DATE and WELL columns.
    well_name : str
        Name/identifier of the well to plot.
    target_col : str, optional
        Production column name to visualize (default: 'W_GAS').
    filepath : str, optional
        Output path for saved plot. If None, generates default name
        based on well name (default: None).

    Returns
    -------
    None
        Saves plot to file or logs warning if well data not found.
    """
    logging.info(f"Generating Time-Series plot for well: {well_name}")

    # Extract data for specified well
    well_data = df[df["WELL"] == well_name].copy()

    if well_data.empty:
        logging.warning(f"No data found for well {well_name}")
        return

    # Create time series plot
    plt.figure(figsize=(14, 6))
    plt.plot(
        well_data["DATE"],
        well_data[target_col],
        label=f"{target_col}",
        color="darkgreen",
        linewidth=2,
    )

    plt.title(f"Production Profile - Well: {well_name}")
    plt.xlabel("Date")
    plt.ylabel(f"Production ({target_col})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Use provided filepath or generate default
    if filepath is None:
        filepath = f"plots/timeseries_{well_name}.png"

    plt.savefig(filepath)
    plt.close()
    logging.info(f"Plot saved to {filepath}")
