"""Train several surrogate model types on one split and compare them."""

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..factory import ModelType, SurrogateFactory

logger = logging.getLogger(__name__)


def benchmark_models(
    df_features: pd.DataFrame,
    qois_dict: dict[Any, Any],
    metadata: dict[str, list[str]],
    model_types: list[ModelType],
    output_dir: str | Path,
    reference_qois: dict[Any, Any] | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    make_plots: bool = True,
    return_factories: bool = False,
    n_pca_components: float | int | None = None,
) -> "pd.DataFrame | tuple[pd.DataFrame, dict[ModelType, SurrogateFactory]]":
    """Train each model type on the **same** split and tabulate physical metrics.

    Using a single ``random_state`` guarantees every model sees the identical
    train/val/test partition, so the comparison is fair.  Per-model plots and a
    pickled model are written under ``output_dir/<MODEL>/`` and the combined
    metrics table is saved as ``benchmark_metrics.csv``.

    Args:
        reference_qois: Optional physical baseline predictions for comparison.
        return_factories: When True, also return the trained factories so the
            caller can reuse them (e.g. for overlay plots) without retraining.

    Returns:
        A metrics table (one row per model/QoI); or, when *return_factories* is
        True, ``(table, {model_type: factory})``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factories: dict[ModelType, SurrogateFactory] = {}

    rows: list[dict[str, Any]] = []
    for model_type in model_types:
        logger.info("Benchmarking model type: %s", model_type.value)
        factory = SurrogateFactory(model_type)
        factory.set_data(df_features.copy(), qois_dict)
        factory.metadata = metadata
        if n_pca_components is not None:
            factory.n_pca_components = n_pca_components
        if reference_qois:
            factory.set_reference_data(reference_qois)

        start = time.perf_counter()
        factory.train(test_size=test_size, val_size=val_size, random_state=random_state)
        train_time = time.perf_counter() - start

        # Inference timing: predict the whole held-out test set.
        infer_start = time.perf_counter()
        test_preds = factory.test_predictions()
        predict_time = time.perf_counter() - infer_start
        n_test_runs = len(next(iter(test_preds.values()))) if test_preds else 0
        predict_time_per_run = predict_time / n_test_runs if n_test_runs else 0.0

        model_dir = output_dir / model_type.value
        summary = factory.evaluate_and_plot(
            plot_prefix=str(model_dir / model_type.value.lower()),
            make_plots=make_plots,
        )
        factory.save_model(model_dir / f"model_{model_type.value}.pkl")
        factories[model_type] = factory

        for qoi_name, metrics in summary.items():
            compressor = factory.compressors_per_qoi.get(qoi_name)
            rows.append(
                {
                    "model": model_type.value,
                    "qoi": qoi_name,
                    "global_r2": metrics["global_r2"],
                    "global_rmse": metrics["global_rmse"],
                    "rmse_mean": metrics["rmse_mean"],
                    "rmse_std": metrics["rmse_std"],
                    "mae_mean": metrics["mae_mean"],
                    "max_err_mean": metrics["max_err_mean"],
                    "nrmse": metrics["nrmse"],
                    "r2_mean": metrics["r2_mean"],
                    "r2_std": metrics["r2_std"],
                    "r2_ref_mean": metrics.get("r2_ref_mean", float("nan")),
                    "n_components": compressor.n_components_ if compressor else None,
                    "explained_variance": (
                        compressor.explained_variance_ratio if compressor else None
                    ),
                    "n_test": metrics["n_samples"],
                    "train_time_s": train_time,
                    "predict_time_s": predict_time,
                    "predict_time_per_run_s": predict_time_per_run,
                }
            )

    table = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_metrics.csv"
    table.to_csv(csv_path, index=False)
    logger.info("Benchmark table written to %s", csv_path)
    if return_factories:
        return table, factories
    return table
