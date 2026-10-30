"""Step 8 temporal deep-learning benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from aeris.evaluation.benchmark import load_ml_ready_dataset
from aeris.evaluation.metrics import mean_absolute_error, phm_score, root_mean_squared_error
from aeris.models.temporal import GRURegressor, LSTMRegressor, TCNRegressor, TransformerRegressor
from aeris.training.datasets import TemporalDataset
from aeris.training.trainer import Trainer, seed_everything, select_device


def _plot_results(
    results: pd.DataFrame,
    histories: dict[str, dict[str, Any]],
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output: Path,
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for metric in ("mae", "rmse", "phm_score", "training_time_seconds", "parameter_count"):
        axis = results.set_index("model")[metric].plot(kind="bar", title=f"{metric} comparison")
        axis.set_ylabel(metric)
        axis.figure.tight_layout()
        axis.figure.savefig(plots / f"{metric}_comparison.png", dpi=150)
        plt.close(axis.figure)
    for name, history in histories.items():
        figure, axis = plt.subplots()
        axis.plot(history["train_loss"], label="train")
        axis.plot(history["validation_loss"], label="validation")
        axis.set(xlabel="Epoch", ylabel="MSE loss", title=f"{name} loss")
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots / f"{name.lower()}_loss.png", dpi=150)
        plt.close(figure)
    best = results[results["model_type"] == "temporal_deep_learning"].sort_values("mae").iloc[0]["model"]
    predicted = predictions[best]
    figure, axis = plt.subplots()
    axis.scatter(y_true, predicted, alpha=0.25, s=10)
    limits = [min(y_true.min(), predicted.min()), max(y_true.max(), predicted.max())]
    axis.plot(limits, limits, "--", color="black")
    axis.set(xlabel="Actual RUL", ylabel="Predicted RUL", title=f"Actual vs predicted RUL: {best}")
    figure.tight_layout()
    figure.savefig(plots / "actual_vs_predicted_best_temporal.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots()
    axis.hist(predicted - y_true, bins=40)
    axis.set(xlabel="Prediction error", ylabel="Count", title=f"Prediction errors: {best}")
    figure.tight_layout()
    figure.savefig(plots / "prediction_error_best_temporal.png", dpi=150)
    plt.close(figure)
    xgboost_path = Path("reports/experiments/step7_classical_ml/predictions/xgboost_predictions.npy")
    if xgboost_path.exists():
        xgboost_predictions = np.load(xgboost_path)
        if xgboost_predictions.shape == y_true.shape:
            figure, axis = plt.subplots(figsize=(9, 4))
            axis.plot(y_true, label="Actual RUL", linewidth=1)
            axis.plot(xgboost_predictions, label="XGBoost", alpha=0.8)
            axis.plot(predicted, label=best, alpha=0.8)
            axis.set(xlabel="Validation sample", ylabel="RUL", title="Actual RUL vs XGBoost vs best temporal model")
            axis.legend()
            figure.tight_layout()
            figure.savefig(plots / "actual_vs_xgboost_vs_best_temporal.png", dpi=150)
            plt.close(figure)


def _write_report(results: pd.DataFrame, output: Path) -> None:
    temporal = results[results["model_type"] == "temporal_deep_learning"]
    best = temporal.sort_values("mae").iloc[0]
    lines = [
        "# Step 8: Temporal Deep Learning Benchmark",
        "",
        "## Objective",
        "Measure whether sequence models improve RUL prediction over the Step 7 classical reference.",
        "",
        "## Dataset and input representation",
        "The leakage-safe FD001 arrays from Step 6 were reused without refitting scalers or regenerating windows.",
        "Temporal models received `(batch, 30, 15)` windows. Classical models use flattened windows.",
        "",
        "## Models and methodology",
        "LSTM, GRU, residual dilated TCN, and compact Transformer Encoder models were trained sequentially",
        "with Adam, MSELoss, gradient clipping, early stopping, and validation-loss checkpointing.",
        "The validation set was used only for evaluation and validation-based checkpoint selection.",
        "",
        "## Results",
        "",
        "```text",
        results.to_string(index=False),
        "```",
        "",
        "## Comparison with XGBoost",
        f"The Step 7 XGBoost reference has MAE {results.loc[results['model'] == 'XGBoost', 'mae'].iloc[0]:.4f}.",
        f"The best temporal model by MAE was {best['model']} with MAE {best['mae']:.4f}.",
        "",
        "## Limitations and conclusion",
        "This is a validation benchmark, not an untouched test estimate. No statistical significance claim is made.",
        "The conclusion should be revisited with an untouched test split and repeated seeds.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_temporal_benchmark(
    data_dir: str | Path = "data/processed/cmapss_fd001",
    output_dir: str | Path = "reports/experiments/step8_temporal_dl",
    seed: int = 42,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    epochs: int = 30,
) -> pd.DataFrame:
    """Train and evaluate all four temporal architectures sequentially."""
    seed_everything(seed)
    X_train, y_train, X_validation, y_validation = load_ml_ready_dataset(data_dir)
    destination = Path(output_dir)
    models_dir = destination / "models"
    predictions_dir = destination / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    train_loader = DataLoader(TemporalDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TemporalDataset(X_validation, y_validation), batch_size=batch_size, shuffle=False)
    builders: dict[str, Callable[[], nn.Module]] = {
        "LSTM": lambda: LSTMRegressor(input_size=X_train.shape[2]),
        "GRU": lambda: GRURegressor(input_size=X_train.shape[2]),
        "TCN": lambda: TCNRegressor(input_size=X_train.shape[2]),
        "Transformer": lambda: TransformerRegressor(input_size=X_train.shape[2], max_length=X_train.shape[1]),
    }
    rows: list[dict[str, Any]] = [{
        "model": "XGBoost",
        "model_type": "classical_reference",
        "mae": 18.698258043515352,
        "rmse": 26.407636014342945,
        "phm_score": 118477.09623510219,
        "training_time_seconds": np.nan,
        "inference_time_seconds": np.nan,
        "inference_time_per_sample": np.nan,
        "parameter_count": np.nan,
        "trainable_parameter_count": np.nan,
        "model_size_mb": np.nan,
        "device": "reference",
    }]
    histories: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, builder in builders.items():
        model = builder()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainer = Trainer(
            model,
            torch.optim.Adam(model.parameters(), lr=learning_rate),
            nn.MSELoss(),
            device,
            models_dir / f"{name.lower()}.pt",
        )
        history = trainer.fit(train_loader, validation_loader, epochs)
        predicted, inference_seconds = trainer.predict(validation_loader)
        predictions[name] = predicted
        histories[name] = history
        np.save(predictions_dir / f"{name.lower()}_predictions.npy", predicted)
        rows.append({
            "model": name,
            "model_type": "temporal_deep_learning",
            "mae": mean_absolute_error(y_validation, predicted),
            "rmse": root_mean_squared_error(y_validation, predicted),
            "phm_score": phm_score(y_validation, predicted),
            "training_time_seconds": history["training_time_seconds"],
            "inference_time_seconds": inference_seconds,
            "inference_time_per_sample": inference_seconds / len(y_validation),
            "parameter_count": parameters,
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
            "device": str(device),
        })
        del model, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results = pd.DataFrame(rows)
    results.to_csv(destination / "results.csv", index=False)
    (destination / "results.json").write_text(json.dumps(results.to_dict(orient="records"), indent=2, default=str), encoding="utf-8")
    (destination / "config.json").write_text(json.dumps({
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "window_size": int(X_train.shape[1]),
        "num_features": int(X_train.shape[2]),
        "optimizer": "Adam",
        "loss": "MSELoss",
        "device": str(device),
    }, indent=2), encoding="utf-8")
    _plot_results(results, histories, y_validation, predictions, destination)
    _write_report(results, destination)
    return results
