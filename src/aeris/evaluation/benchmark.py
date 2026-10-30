"""End-to-end classical ML benchmark for the prepared FD001 arrays."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aeris.evaluation.metrics import mean_absolute_error, phm_score, root_mean_squared_error
from aeris.models.classical import build_classical_models, flatten_temporal_windows


def load_ml_ready_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the four Step 6 ML-ready arrays."""
    directory = Path(data_dir)
    arrays = tuple(np.load(directory / name) for name in (
        "X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy"
    ))
    X_train, y_train, X_validation, y_validation = arrays
    validate_ml_ready_dataset(X_train, y_train, X_validation, y_validation)
    return X_train, y_train, X_validation, y_validation


def validate_ml_ready_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> None:
    """Validate dimensions, row alignment, and finite numeric values."""
    for name, values in (("X_train", X_train), ("X_validation", X_validation)):
        if np.asarray(values).ndim != 3:
            raise ValueError(f"{name} must be a 3-D temporal array.")
        if not np.issubdtype(np.asarray(values).dtype, np.number):
            raise ValueError(f"{name} must contain numeric values.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")
    for name, values, expected_rows in (
        ("y_train", y_train, len(X_train)),
        ("y_validation", y_validation, len(X_validation)),
    ):
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != expected_rows:
            raise ValueError(f"{name} must be 1-D and aligned with its feature array.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")
    if X_train.shape[1:] != X_validation.shape[1:]:
        raise ValueError("Training and validation windows must have the same shape.")


def _save_plots(results: pd.DataFrame, y_true: np.ndarray, predictions: dict[str, np.ndarray], output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric, filename in (("mae", "mae_comparison.png"), ("rmse", "rmse_comparison.png")):
        ax = results.set_index("model")[metric].plot(kind="bar", title=f"{metric.upper()} comparison")
        ax.set_ylabel(metric.upper())
        ax.figure.tight_layout()
        ax.figure.savefig(plots_dir / filename, dpi=150)
        plt.close(ax.figure)

    best_model = results.sort_values("mae").iloc[0]["model"]
    best_predictions = predictions[best_model]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(y_true, best_predictions, alpha=0.25, s=10)
    limits = [min(y_true.min(), best_predictions.min()), max(y_true.max(), best_predictions.max())]
    axis.plot(limits, limits, "--", color="black")
    axis.set(xlabel="Actual RUL", ylabel="Predicted RUL", title=f"Actual vs predicted RUL: {best_model}")
    figure.tight_layout()
    figure.savefig(plots_dir / "actual_vs_predicted_best_model.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(best_predictions - y_true, bins=40)
    axis.set(xlabel="Prediction error (predicted - actual)", ylabel="Count", title=f"Prediction errors: {best_model}")
    figure.tight_layout()
    figure.savefig(plots_dir / "prediction_error_distribution.png", dpi=150)
    plt.close(figure)


def run_benchmark(
    data_dir: str | Path = "data/processed/cmapss_fd001",
    output_dir: str | Path = "reports/experiments/step7_classical_ml",
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit each baseline on training data and evaluate only on validation data."""
    X_train, y_train, X_validation, y_validation = load_ml_ready_dataset(data_dir)
    flattened_train = flatten_temporal_windows(X_train)
    flattened_validation = flatten_temporal_windows(X_validation)
    destination = Path(output_dir)
    models_dir = destination / "models"
    predictions_dir = destination / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in build_classical_models(random_state).items():
        started = time.perf_counter()
        model.fit(flattened_train, y_train)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted = np.asarray(model.predict(flattened_validation), dtype=float)
        inference_seconds = time.perf_counter() - started
        predictions[name] = predicted
        np.save(predictions_dir / f"{name}_predictions.npy", predicted)
        joblib.dump(model, models_dir / f"{name}.joblib")
        results.append({
            "model": name,
            "mae": mean_absolute_error(y_validation, predicted),
            "rmse": root_mean_squared_error(y_validation, predicted),
            "phm_score": phm_score(y_validation, predicted),
            "training_time_seconds": training_seconds,
            "inference_time_seconds": inference_seconds,
        })

    frame = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    frame.to_csv(destination / "comparison_results.csv", index=False)
    (destination / "comparison_results.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    _save_plots(frame, y_validation, predictions, destination)
    return frame
