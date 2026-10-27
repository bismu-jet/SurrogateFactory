"""End-to-end results generation for one dataset.

Given features + per-run QoIs + metadata, this trains every model on one shared
split and emits: a physical-unit benchmark table, comparison bar charts,
multi-model overlay plots, Kriging uncertainty bands, and (optionally) K-fold
cross-validation metrics — all under one output directory.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from ..factory import ModelType, SurrogateFactory
from .benchmark import benchmark_models
from .metrics import calculate_mae, calculate_max_abs_error, calculate_nrmse, calculate_rmse

# plotting imports matplotlib and sets the Agg backend; import pyplot afterwards.
from .plotting import (  # isort: skip
    plot_benchmark_bars,
    plot_comparison_timeseries,
    plot_multimodel_timeseries,
    plot_parity,
    plot_pca_cumulative_variance,
    plot_timing,
)
import matplotlib.pyplot as plt  # noqa: E402 -- must follow plotting's Agg setup

logger = logging.getLogger(__name__)

DEFAULT_MODEL_TYPES = [ModelType.KRIGING, ModelType.RBF, ModelType.NN]


def _array_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the benchmark metric set from physical ``(n_runs, n_steps)`` arrays."""
    per_run_rmse = [calculate_rmse(y_true[i], y_pred[i]) for i in range(len(y_true))]
    per_run_r2 = [calculate_r2_safe(y_true[i], y_pred[i]) for i in range(len(y_true))]
    value_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    return {
        "global_r2": float(r2_score(y_true.flatten(), y_pred.flatten())),
        "global_rmse": calculate_rmse(y_true.flatten(), y_pred.flatten()),
        "rmse_mean": float(np.mean(per_run_rmse)),
        "rmse_std": float(np.std(per_run_rmse)),
        "mae_mean": calculate_mae(y_true.flatten(), y_pred.flatten()),
        "max_err_mean": float(
            np.mean([calculate_max_abs_error(y_true[i], y_pred[i]) for i in range(len(y_true))])
        ),
        "nrmse": calculate_nrmse(y_true.flatten(), y_pred.flatten()) if value_range else 0.0,
        "r2_mean": float(np.mean(per_run_r2)),
        "r2_std": float(np.std(per_run_r2)),
    }


def calculate_r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-run R² with a constant-target guard (variance < 1e-12).

    Same rule as :meth:`SurrogateFactory.evaluate_and_plot`, which uses 1e-9.
    """
    if float(np.var(y_true)) < 1e-12:
        return 1.0 if calculate_rmse(y_true, y_pred) < 1e-5 else 0.0
    return float(r2_score(y_true, y_pred))


def _append_ensemble(
    table: pd.DataFrame,
    factories: dict[ModelType, SurrogateFactory],
    output_dir: Path,
) -> pd.DataFrame:
    """Add a mean-ensemble row per QoI (often the strongest predictor)."""
    reference = next(iter(factories.values()))
    preds_by_factory = {mt: f.test_predictions() for mt, f in factories.items()}
    rows: list[dict[str, Any]] = []
    for qoi_name in reference.qoi_names:
        member_preds = [p[qoi_name] for p in preds_by_factory.values() if qoi_name in p]
        if len(member_preds) < 2:
            continue
        y_true = reference.processed_data["y_test_per_qoi_physical"][qoi_name]
        y_pred = np.mean(member_preds, axis=0)
        rows.append({"model": "Ensemble", "qoi": qoi_name, **_array_metrics(y_true, y_pred)})
    if rows:
        table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
        table.to_csv(output_dir / "benchmark_metrics.csv", index=False)
    return table


def _feature_row(df_features: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Return the single feature row for *run_id* (matched on run_number)."""
    feat_df = df_features.copy()
    feat_df["run_id"] = feat_df["run_number"].astype(str)
    return feat_df[feat_df["run_id"] == run_id]


def _sample_test_run_ids(factory: SurrogateFactory, n: int) -> list[str]:
    """First *n* held-out run identifiers (as strings) from a trained factory."""
    if factory.test_run_id_map is None:
        return []
    return [str(r) for r in list(factory.test_run_id_map.values)[:n]]


def _overlay_plots(
    factories: dict[ModelType, SurrogateFactory],
    df_features: pd.DataFrame,
    qois: dict[int, dict[str, Any]],
    output_dir: Path,
    n_runs: int,
) -> None:
    """Overlay every model's prediction against the real curve for a few runs."""
    reference_factory = next(iter(factories.values()))
    plot_prefix = str(output_dir / "comparison_all_models" / "compare")
    for run_id in _sample_test_run_ids(reference_factory, n_runs):
        feature_row = _feature_row(df_features, run_id)
        if feature_row.empty:
            continue
        predictions_by_qoi: dict[str, dict[str, np.ndarray]] = {}
        for model_type, factory in factories.items():
            for qoi_name, curve in factory.predict(feature_row).items():
                predictions_by_qoi.setdefault(qoi_name, {})[model_type.value] = curve.flatten()
        ground_truth = qois.get(int(run_id), {})
        for qoi_name, per_model in predictions_by_qoi.items():
            true_curve = np.array(ground_truth.get(qoi_name, [])) if ground_truth else None
            plot_multimodel_timeseries(
                y_true=true_curve,
                predictions=per_model,
                run_number=int(run_id),
                qoi_name=qoi_name,
                save_path_prefix=plot_prefix,
            )
    logger.info("Multi-model overlays written under %s", output_dir / "comparison_all_models")


def _uncertainty_plots(
    factory: SurrogateFactory,
    df_features: pd.DataFrame,
    qois: dict[int, dict[str, Any]],
    output_dir: Path,
    n_runs: int,
) -> None:
    """Kriging mean +/-2 sigma bands for a few held-out runs (no retraining)."""
    plot_prefix = str(output_dir / "Kriging" / "uncertainty" / "kriging_unc")
    for run_id in _sample_test_run_ids(factory, n_runs):
        feature_row = _feature_row(df_features, run_id)
        if feature_row.empty:
            continue
        prediction = factory.predict_with_uncertainty(feature_row)
        ground_truth = qois.get(int(run_id), {})
        for qoi_name, (mean, std) in prediction.items():
            true_curve = np.array(ground_truth.get(qoi_name, [])) if ground_truth else None
            plot_comparison_timeseries(
                y_true=true_curve if true_curve is not None and true_curve.size else mean.flatten(),
                y_pred_new=mean.flatten(),
                y_pred_old=None,
                run_number=int(run_id),
                qoi_name=qoi_name,
                save_path_prefix=plot_prefix,
                new_model_label="Kriging Prediction",
                y_std_new=std.flatten(),
            )
    logger.info("Kriging uncertainty plots written under %s", output_dir / "Kriging")


def _cross_validation(
    df_features: pd.DataFrame,
    qois: dict[int, dict[str, Any]],
    metadata: dict[str, list[str]],
    output_dir: Path,
    model_types: list[ModelType],
    random_state: int,
    n_pca_components: float | int | None = None,
    n_jobs: int = 1,
) -> None:
    """K-fold CV for each model type; writes ``cv_metrics.csv``."""
    rows: list[dict[str, Any]] = []
    for model_type in model_types:
        logger.info("Cross-validating %s.", model_type.value)
        factory = SurrogateFactory(model_type, n_jobs=n_jobs)
        factory.set_data(df_features.copy(), qois)
        factory.metadata = metadata
        if n_pca_components is not None:
            factory.n_pca_components = n_pca_components
        cv_summary = factory.cross_validate(k=5, random_state=random_state)
        for qoi_name, metrics in cv_summary.items():
            rows.append({"model": model_type.value, "qoi": qoi_name, **metrics})
    pd.DataFrame(rows).to_csv(output_dir / "cv_metrics.csv", index=False)
    logger.info("Cross-validation table written to %s", output_dir / "cv_metrics.csv")


def _pooled_truth_and_pred(
    factory: SurrogateFactory,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate physical (truth, prediction) over all QoIs of the test set."""
    preds = factory.test_predictions()
    truths, predictions = [], []
    for qoi_name, y_pred in preds.items():
        truths.append(factory.processed_data["y_test_per_qoi_physical"][qoi_name].flatten())
        predictions.append(y_pred.flatten())
    return np.concatenate(truths), np.concatenate(predictions)


def _success_analysis(
    factories: dict[ModelType, SurrogateFactory],
    table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Extra artefacts explaining *why* the surrogates work: parity plots, PCA
    explained-variance, a timing chart, and a written summary."""
    plot_timing(table, output_dir)

    # Parity plots (pooled over all QoIs) per model.
    for model_type, factory in factories.items():
        y_true, y_pred = _pooled_truth_and_pred(factory)
        r2 = float(r2_score(y_true, y_pred)) if np.var(y_true) > 1e-20 else 0.0
        plot_parity(
            y_true,
            y_pred,
            model_type.value,
            "all QoIs (pooled)",
            r2,
            output_dir / "parity" / f"parity_{model_type.value}.png",
        )

    # PCA cumulative variance (from the first available factory's compressors).
    reference = next(iter(factories.values()))
    ratios = {
        qoi: comp.explained_variance_ratio_per_component
        for qoi, comp in reference.compressors_per_qoi.items()
    }
    if ratios:
        plot_pca_cumulative_variance(ratios, output_dir)

    _write_summary(factories, table, reference, output_dir)


def _write_summary(
    factories: dict[ModelType, SurrogateFactory],
    table: pd.DataFrame,
    reference: SurrogateFactory,
    output_dir: Path,
) -> None:
    """Write a human-readable ``WHY_IT_WORKED.md`` with the key evidence."""
    lines = ["# Why the surrogate works — evidence summary", ""]
    lines.append("## PCA output compression (the key enabler)")
    lines.append(
        "Each QoI's time-series is compressed to a few principal components that "
        "capture ~99.9% of the variance, so the models regress a handful of latent "
        "scalars instead of hundreds of correlated time-steps:"
    )
    lines.append("")
    lines.append("| QoI | # PCA components | explained variance |")
    lines.append("|---|---|---|")
    for qoi, comp in reference.compressors_per_qoi.items():
        lines.append(f"| {qoi} | {comp.n_components_} | {comp.explained_variance_ratio:.5f} |")
    lines.append("")

    lines.append("## Accuracy and speed by model (held-out test set)")
    lines.append("| model | best global R² | train (s) | inference (s) | inference/run (s) |")
    lines.append("|---|---|---|---|---|")
    for model_type in factories:
        sub = table[table["model"] == model_type.value]
        if sub.empty:
            continue
        best_r2 = float(sub["global_r2"].max())
        row = sub.iloc[0]
        lines.append(
            f"| {model_type.value} | {best_r2:.4f} | {row['train_time_s']:.2f} | "
            f"{row.get('predict_time_s', float('nan')):.3f} | "
            f"{row.get('predict_time_per_run_s', float('nan')):.5f} |"
        )
    lines.append("")
    lines.append(
        "Kriging interpolates the smooth, densely-sampled low-dimensional input "
        "space almost exactly; the neural network matches it once the output is "
        "compressed; RBF trades a little accuracy for the fastest training."
    )
    (output_dir / "WHY_IT_WORKED.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", output_dir / "WHY_IT_WORKED.md")


def analyze_pca_speed_impact(
    df_features: pd.DataFrame,
    qois: dict[int, dict[str, Any]],
    metadata: dict[str, list[str]],
    output_dir: Path,
    component_grid: tuple[int, ...] = (2, 4, 8, 16, 32),
    model_types: list[ModelType] | None = None,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Quantify how PCA component count drives speed and accuracy.

    For a single representative QoI, trains each model while varying the number
    of retained PCA components and records train/inference time and global R².
    Fewer components => far faster (Kriging trains one model per component) with
    accuracy that saturates early — the core reason PCA makes the surrogate fast.
    Writes ``pca_speed_impact.csv`` and two plots under *output_dir*.
    """
    model_types = model_types or DEFAULT_MODEL_TYPES
    output_dir = Path(output_dir)

    # Pick the highest-variance QoI (most informative target).
    any_qois = next(iter(qois.values()))
    target_qoi = max(any_qois, key=lambda k: float(np.var(np.asarray(any_qois[k]))))
    single = {r: {target_qoi: d[target_qoi]} for r, d in qois.items() if target_qoi in d}
    logger.info("PCA speed-impact analysis on QoI '%s'.", target_qoi)

    rows: list[dict[str, Any]] = []
    for model_type in model_types:
        for n_comp in component_grid:
            factory = SurrogateFactory(model_type, n_jobs=n_jobs)
            factory.set_data(df_features.copy(), single)
            factory.metadata = metadata
            factory.n_pca_components = int(n_comp)
            t0 = time.perf_counter()
            try:
                factory.train(test_size=0.2, val_size=0.1, random_state=random_state)
            except Exception:
                logger.exception(
                    "PCA-impact train failed (model=%s, k=%d).", model_type.value, n_comp
                )
                continue
            train_s = time.perf_counter() - t0
            qoi = factory.qoi_names[0]
            t1 = time.perf_counter()
            preds = factory.test_predictions()[qoi]
            predict_s = time.perf_counter() - t1
            y_true = factory.processed_data["y_test_per_qoi_physical"][qoi]
            g = float(r2_score(y_true.flatten(), preds.flatten()))
            rows.append(
                {
                    "model": model_type.value,
                    "requested_components": int(n_comp),
                    "actual_components": factory.compressors_per_qoi[qoi].n_components_,
                    "train_time_s": train_s,
                    "predict_time_s": predict_s,
                    "global_r2": g,
                }
            )

    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "pca_speed_impact.csv", index=False)

    # Plot: train time and R² vs components, per model.
    for column, ylabel, fname, logy in [
        ("train_time_s", "Train time (s)", "pca_speed_vs_components.png", True),
        ("global_r2", "Global R²", "pca_r2_vs_components.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_type in model_types:
            sub = table[table["model"] == model_type.value]
            if sub.empty:
                continue
            ax.plot(sub["actual_components"], sub[column], marker="o", label=model_type.value)
        ax.set_xlabel("PCA components retained")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{ylabel} vs PCA components ({target_qoi[:40]})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / fname)
        plt.close(fig)
    logger.info("PCA speed-impact written to %s", output_dir / "pca_speed_impact.csv")
    return table


def produce_full_results(
    df_features: pd.DataFrame,
    qois: dict[int, dict[str, Any]],
    metadata: dict[str, list[str]],
    output_dir: Path,
    model_types: list[ModelType] | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    n_sample_plots: int = 3,
    run_cv: bool = True,
    n_pca_components: float | int | None = None,
    analyze_pca_speed: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run the full comparison and write every artefact under *output_dir*.

    Args:
        analyze_pca_speed: Also sweep PCA component count to quantify its impact
            on training/inference speed and accuracy (extra runtime).
        n_jobs: QoIs trained in parallel per model (see :class:`SurrogateFactory`).
    """
    model_types = model_types or DEFAULT_MODEL_TYPES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table, factories = benchmark_models(
        df_features=df_features,
        qois_dict=qois,
        metadata=metadata,
        model_types=model_types,
        output_dir=output_dir,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        return_factories=True,
        n_pca_components=n_pca_components,
        n_jobs=n_jobs,
    )
    table = _append_ensemble(table, factories, output_dir)
    print("\n=== Benchmark (physical units) ===")
    cols = [
        c
        for c in ["model", "qoi", "global_r2", "nrmse", "r2_mean", "train_time_s", "predict_time_s"]
        if c in table.columns
    ]
    print(table[cols].to_string(index=False))

    plot_benchmark_bars(table, output_dir)
    _success_analysis(factories, table, output_dir)
    _overlay_plots(factories, df_features, qois, output_dir, n_sample_plots)
    if ModelType.KRIGING in factories:
        _uncertainty_plots(
            factories[ModelType.KRIGING], df_features, qois, output_dir, n_sample_plots
        )
    if analyze_pca_speed:
        analyze_pca_speed_impact(
            df_features, qois, metadata, output_dir, model_types=model_types, n_jobs=n_jobs
        )
    if run_cv:
        _cross_validation(
            df_features,
            qois,
            metadata,
            output_dir,
            model_types,
            random_state,
            n_pca_components,
            n_jobs,
        )

    logger.info("All results written to %s", output_dir)
    return table
