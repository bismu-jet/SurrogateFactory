"""Visualisation utilities for surrogate model evaluation."""

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_feature_distribution(
    x_data: pd.DataFrame,
    save_path: str = "feature_distribution.png",
) -> None:
    """Scatter-plot of training samples in feature space (best for 2-D)."""
    if x_data.shape[1] != 2:
        logger.warning(
            "Feature distribution plot works best with exactly 2 features; got %d.",
            x_data.shape[1],
        )
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(x_data.iloc[:, 0], x_data.iloc[:, 1], alpha=0.7)
    plt.title("Training Sample Distribution in Feature Space")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info("Feature distribution plot saved to %s", save_path)


def plot_target_timeseries(
    y_data: np.ndarray,
    num_to_plot: int = 5,
    save_path: str = "target_timeseries.png",
) -> None:
    """Plot a handful of target time-series curves."""
    plt.figure(figsize=(12, 6))
    num_to_plot = min(num_to_plot, len(y_data))
    for i in range(num_to_plot):
        plt.plot(y_data[i], label=f"Sample {i + 1}", alpha=0.8)
    plt.title(f"First {num_to_plot} Target Time-Series (y_train)")
    plt.xlabel("Time-Step")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info("Target time-series plot saved to %s", save_path)


def plot_comparison_timeseries(
    y_true: np.ndarray,
    y_pred_new: np.ndarray,
    y_pred_old: Optional[np.ndarray],
    run_number: int,
    qoi_name: str,
    save_path_prefix: str = "comparison",
    new_model_label: str = "New Model Prediction",
) -> None:
    """Plot ground-truth vs. predicted time-series for a single run/QoI."""
    plt.figure(figsize=(15, 7))
    plt.plot(y_true, label="Ground Truth", color="black", linewidth=2.5, alpha=0.8)
    plt.plot(y_pred_new, label=new_model_label, color="blue", linestyle="--", linewidth=2)
    if y_pred_old is not None:
        plt.plot(
            y_pred_old, label="Reference Model", color="red", linestyle=":", linewidth=2
        )

    plt.title(f"Model Comparison – Run #{run_number}\nQoI: {qoi_name}")
    plt.xlabel("Concatenated Time-Steps")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    qoi_safe = (
        str(qoi_name)
        .replace(" ", "_")
        .replace(":", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("(", "")
        .replace(")", "")
    )
    final_save_path = f"{save_path_prefix}_run_{run_number}_qoi_{qoi_safe}.png"
    os.makedirs(os.path.dirname(final_save_path) or ".", exist_ok=True)
    plt.savefig(final_save_path)
    plt.close()
    logger.info("Comparison plot for run #%d (%s) saved to %s", run_number, qoi_name, final_save_path)
