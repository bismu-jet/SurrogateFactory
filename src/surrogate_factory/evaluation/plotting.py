"""Visualisation utilities for surrogate model evaluation."""

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent per-model styling across all comparison plots.
_MODEL_STYLES = {
    "Kriging": ("--", "tab:blue"),
    "RBF": (":", "tab:red"),
    "NeuralNetwork": ("-.", "tab:green"),
}


def _short_qoi(qoi_name: str) -> str:
    """Drop the trailing ``- project...`` study id for compact axis labels."""
    return qoi_name.split(" - project")[0]


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
    y_pred_old: np.ndarray | None,
    run_number: int,
    qoi_name: str,
    save_path_prefix: str = "comparison",
    new_model_label: str = "New Model Prediction",
    y_std_new: np.ndarray | None = None,
) -> None:
    """Plot ground-truth vs. predicted time-series for a single run/QoI.

    When *y_std_new* is given (e.g. Kriging uncertainty), a +/-2 sigma band is
    shaded around the new model's prediction.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(y_true, label="Ground Truth", color="black", linewidth=2.5, alpha=0.8)
    plt.plot(y_pred_new, label=new_model_label, color="blue", linestyle="--", linewidth=2)
    if y_std_new is not None:
        steps = np.arange(len(y_pred_new))
        plt.fill_between(
            steps,
            np.ravel(y_pred_new) - 2.0 * np.ravel(y_std_new),
            np.ravel(y_pred_new) + 2.0 * np.ravel(y_std_new),
            color="blue",
            alpha=0.15,
            label="+/-2 sigma",
        )
    if y_pred_old is not None:
        plt.plot(y_pred_old, label="Reference Model", color="red", linestyle=":", linewidth=2)

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
    logger.info(
        "Comparison plot for run #%d (%s) saved to %s", run_number, qoi_name, final_save_path
    )


def plot_multimodel_timeseries(
    y_true: np.ndarray | None,
    predictions: dict[str, np.ndarray],
    run_number: int,
    qoi_name: str,
    save_path_prefix: str = "compare",
) -> None:
    """Overlay every model's predicted curve against the ground truth for one run/QoI.

    Unlike :func:`plot_comparison_timeseries` (one model per figure), this puts
    the real series and all surrogate predictions on the same axes so the
    methods can be compared directly.
    """
    plt.figure(figsize=(15, 7))
    if y_true is not None and len(y_true):
        plt.plot(y_true, label="Ground Truth", color="black", linewidth=2.5, alpha=0.85)
    for model_name, curve in predictions.items():
        linestyle, color = _MODEL_STYLES.get(model_name, ("--", "tab:gray"))
        plt.plot(np.ravel(curve), label=model_name, linestyle=linestyle, color=color, linewidth=1.8)

    plt.title(f"All models vs ground truth – Run #{run_number}\nQoI: {qoi_name}")
    plt.xlabel("Concatenated Time-Steps")
    plt.ylabel("Output Value (physical units)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    qoi_safe = _short_qoi(qoi_name).replace(" ", "_").replace("/", "-")
    final_save_path = f"{save_path_prefix}_run_{run_number}_qoi_{qoi_safe}.png"
    os.makedirs(os.path.dirname(final_save_path) or ".", exist_ok=True)
    plt.savefig(final_save_path)
    plt.close()


def plot_benchmark_bars(table: pd.DataFrame, output_dir: str | Path) -> None:
    """Grouped bar charts comparing the models per QoI, plus training time.

    Writes ``bench_rmse_mean.png``, ``bench_nrmse.png``, ``bench_r2_mean.png``
    and ``bench_train_time.png`` under *output_dir*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = table.copy()
    data["qoi_short"] = data["qoi"].map(_short_qoi)
    models = list(dict.fromkeys(data["model"]))
    qois = list(dict.fromkeys(data["qoi_short"]))
    x = np.arange(len(qois))
    width = 0.8 / max(len(models), 1)

    bar_metrics = [
        ("global_r2", "Global R² (pooled)"),
        ("rmse_mean", "RMSE (physical units)"),
        ("nrmse", "NRMSE (fraction of range)"),
        ("r2_mean", "R² (per-run mean)"),
    ]
    for column, label in [bm for bm in bar_metrics if bm[0] in data.columns]:
        fig, ax = plt.subplots(figsize=(max(8.0, len(qois) * 1.8), 6))
        for i, model_name in enumerate(models):
            subset = data[data["model"] == model_name].set_index("qoi_short").reindex(qois)
            ax.bar(x + i * width, subset[column].to_numpy(), width, label=model_name)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(qois, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by QoI and model")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / f"bench_{column}.png")
        plt.close(fig)

    logger.info("Benchmark bar charts written under %s", output_dir)


def plot_timing(table: pd.DataFrame, output_dir: str | Path) -> None:
    """Grouped bars of training and inference time per model (log scale)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # One row per model (timing is per model, repeated across QoIs).
    per_model = table.drop_duplicates(subset="model").set_index("model")
    models = list(per_model.index)
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.38
    ax.bar(x - width / 2, per_model["train_time_s"].to_numpy(), width, label="Train")
    if "predict_time_s" in per_model.columns:
        ax.bar(
            x + width / 2,
            per_model["predict_time_s"].to_numpy(),
            width,
            label="Inference (test set)",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Time (s, log scale)")
    ax.set_yscale("log")
    ax.set_title("Processing time by model")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    # Annotate train bars with values.
    for i in range(len(models)):
        ax.annotate(
            f"{per_model['train_time_s'].iloc[i]:.2f}s",
            (x[i] - width / 2, per_model["train_time_s"].iloc[i]),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "timing_by_model.png")
    plt.close(fig)
    logger.info("Timing chart written to %s", output_dir / "timing_by_model.png")


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    qoi_name: str,
    r2: float,
    save_path: str | Path,
) -> None:
    """Predicted-vs-actual scatter with the ideal y=x line (a fit-quality view)."""
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=8, alpha=0.3, color="tab:blue")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal (y = x)")
    ax.set_xlabel("Ground truth (physical)")
    ax.set_ylabel("Prediction (physical)")
    ax.set_title(f"Parity – {model_name}\n{_short_qoi(qoi_name)}  (R² = {r2:.4f})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_pca_cumulative_variance(
    ratios_per_qoi: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    """Cumulative PCA explained-variance per QoI — shows why compression works.

    A sharp knee (few components reaching ~99.9%) is the reason the latent
    regression is easy and the surrogate is accurate.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for qoi_name, ratios in ratios_per_qoi.items():
        cumulative = np.cumsum(ratios)
        components = np.arange(1, len(cumulative) + 1)
        ax.plot(
            components, cumulative, marker="o", markersize=3, alpha=0.7, label=_short_qoi(qoi_name)
        )
    ax.axhline(0.999, color="red", linestyle="--", linewidth=1, label="99.9%")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA compression: cumulative explained variance per QoI")
    ax.grid(True, linestyle="--", alpha=0.4)
    if len(ratios_per_qoi) <= 12:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_cumulative_variance.png")
    plt.close(fig)
    logger.info("PCA variance plot written to %s", output_dir / "pca_cumulative_variance.png")
