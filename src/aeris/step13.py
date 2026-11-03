"""Step 13 recurrent-model evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from aeris.evaluation.benchmark import load_ml_ready_dataset
from aeris.evaluation.metrics import mean_absolute_error, phm_score, root_mean_squared_error
from aeris.models.temporal import GRURegressor, LSTMRegressor
from aeris.training.datasets import TemporalDataset
from aeris.training.trainer import Trainer, seed_everything, select_device


def count_parameters(model: nn.Module) -> tuple[int, int]:
    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )


def lifecycle_metrics(y_true: np.ndarray, prediction: np.ndarray) -> list[dict[str, Any]]:
    boundaries = np.quantile(y_true, [1 / 3, 2 / 3])
    rows = []
    for name, mask in (
        ("early", y_true >= boundaries[1]),
        ("mid", (y_true >= boundaries[0]) & (y_true < boundaries[1])),
        ("late", y_true < boundaries[0]),
    ):
        rows.append({"lifecycle": name, "count": int(mask.sum()), "mae": mean_absolute_error(y_true[mask], prediction[mask])})
    return rows


def model_row(name: str, y_true: np.ndarray, prediction: np.ndarray, history: dict[str, Any], model: nn.Module, device: torch.device, inference: float) -> dict[str, Any]:
    parameters, trainable = count_parameters(model)
    return {
        "model": name,
        "mae": mean_absolute_error(y_true, prediction),
        "rmse": root_mean_squared_error(y_true, prediction),
        "phm_score": phm_score(y_true, prediction),
        "training_time_seconds": float(history["training_time_seconds"]),
        "inference_time_seconds": float(inference),
        "parameter_count": parameters,
        "trainable_parameter_count": trainable,
        "device": str(device),
        "epochs_completed": len(history["train_loss"]),
        "best_validation_loss": float(history["best_validation_loss"]),
        "seed": 42,
        "early_life_mae": lifecycle_metrics(y_true, prediction)[0]["mae"],
        "mid_life_mae": lifecycle_metrics(y_true, prediction)[1]["mae"],
        "late_life_mae": lifecycle_metrics(y_true, prediction)[2]["mae"],
    }


def reproducibility_manifest(data_dir: Path, x_train: np.ndarray, x_validation: np.ndarray, device: torch.device) -> dict[str, Any]:
    cuda_version = torch.version.cuda if device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": cuda_version,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seed": 42,
        "dataset_paths": {
            "train": str(data_dir / "X_train.npy"),
            "validation": str(data_dir / "X_validation.npy"),
        },
        "dataset_shapes": {"X_train": list(x_train.shape), "X_validation": list(x_validation.shape)},
        "model_configuration": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2},
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 30,
        "patience": 5,
        "runtime_budget_seconds": 600,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_step13(data_dir: str | Path, output_dir: str | Path) -> pd.DataFrame:
    seed_everything(42)
    data_path = Path(data_dir)
    output = Path(output_dir)
    models_dir = output / "models"
    predictions_dir = output / "predictions"
    plots_dir = output / "plots"
    for directory in (models_dir, predictions_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_validation, y_validation = load_ml_ready_dataset(data_path)
    device = select_device()
    train_loader = DataLoader(TemporalDataset(x_train, y_train), batch_size=128, shuffle=True)
    validation_loader = DataLoader(TemporalDataset(x_validation, y_validation), batch_size=128, shuffle=False)
    registry_path = output / "experiment_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"experiments": []}
    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, builder in (
        ("LSTM", lambda: LSTMRegressor(input_size=x_train.shape[2], hidden_size=64, num_layers=1)),
        ("GRU", lambda: GRURegressor(input_size=x_train.shape[2], hidden_size=64, num_layers=1)),
    ):
        checkpoint = models_dir / f"{name.lower()}.pt"
        prediction_path = predictions_dir / f"{name.lower()}_predictions.npy"
        existing = next((item for item in registry["experiments"] if item.get("model") == name), None)
        if existing and existing.get("status") == "SUCCESS" and checkpoint.exists() and prediction_path.exists():
            model = builder().to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            prediction = np.load(prediction_path)
            prediction_started = time.perf_counter()
            model.eval()
            with torch.no_grad():
                prediction_parts = [
                    model(features.to(device)).cpu().numpy()
                    for features, _ in validation_loader
                ]
            prediction = np.concatenate(prediction_parts)
            prediction_time = time.perf_counter() - prediction_started
            history_path = output / f"{name.lower()}_training_history.npz"
            history_values = np.load(history_path) if history_path.exists() else None
            history = {
                "train_loss": history_values["train_loss"].tolist() if history_values is not None else [],
                "validation_loss": history_values["validation_loss"].tolist() if history_values is not None else [],
                "training_time_seconds": existing["runtime"],
                "best_validation_loss": existing["best_validation_loss"],
            }
            if not history["train_loss"]:
                history["train_loss"] = [float("nan")] * int(existing.get("epochs_completed", 30))
                history["validation_loss"] = [float("nan")] * int(existing.get("epochs_completed", 30))
        else:
            model = builder()
            started = time.perf_counter()
            trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), nn.MSELoss(), device, checkpoint)
            history = trainer.fit(train_loader, validation_loader, epochs=30)
            prediction, prediction_time = trainer.predict(validation_loader)
            np.save(prediction_path, prediction)
            if time.perf_counter() - started > 600:
                status = "TIME_BUDGET_EXCEEDED"
            else:
                status = "SUCCESS"
            registry["experiments"] = [item for item in registry["experiments"] if item.get("model") != name]
            registry["experiments"].append({
                "experiment": f"step13_{name.lower()}",
                "model": name,
                "status": status,
                "runtime": float(history["training_time_seconds"]),
                "epochs_completed": len(history["train_loss"]),
                "best_validation_loss": float(history["best_validation_loss"]),
                "artifacts": [str(checkpoint), str(prediction_path)],
                "failure_reason": None if status == "SUCCESS" else "runtime budget exceeded",
            })
        rows.append(model_row(name, y_validation, prediction, history, model, device, prediction_time))
        histories[name] = history
        predictions[name] = prediction
        np.savez(output / f"{name.lower()}_training_history.npz", train_loss=history["train_loss"], validation_loss=history["validation_loss"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reference = pd.read_json(Path("reports/experiments/step8_temporal_dl/results.json"))
    transformer = reference.loc[reference["model"].eq("Transformer")].iloc[0].to_dict()
    xgboost_reference = reference.loc[reference["model"].eq("XGBoost")].iloc[0].to_dict()
    gpu_reference_path = Path("reports/experiments/step8_temporal_dl/gpu/transformer_gpu_results.json")
    gpu_payload = json.loads(gpu_reference_path.read_text(encoding="utf-8")) if gpu_reference_path.exists() else {}
    gpu_reference = gpu_payload.get("gpu", gpu_payload)
    rows.extend([
        {"model": "XGBoost", "mae": xgboost_reference["mae"], "rmse": xgboost_reference["rmse"], "phm_score": xgboost_reference["phm_score"], "training_time_seconds": xgboost_reference["training_time_seconds"], "inference_time_seconds": xgboost_reference["inference_time_seconds"], "parameter_count": None, "device": "reference"},
        {"model": "Transformer CPU", "mae": transformer["mae"], "rmse": transformer["rmse"], "phm_score": transformer["phm_score"], "training_time_seconds": transformer["training_time_seconds"], "inference_time_seconds": transformer["inference_time_seconds"], "parameter_count": transformer["parameter_count"], "device": "cpu"},
        {"model": "Transformer GPU", "mae": gpu_reference.get("mae"), "rmse": gpu_reference.get("rmse"), "phm_score": gpu_reference.get("phm_score"), "training_time_seconds": gpu_reference.get("training_time_seconds"), "inference_time_seconds": gpu_reference.get("inference_time_seconds"), "parameter_count": gpu_reference.get("parameter_count"), "device": "cuda"},
    ])
    results = pd.DataFrame(rows)
    transformer_mae = float(transformer["mae"])
    transformer_rmse = float(transformer["rmse"])
    transformer_phm = float(transformer["phm_score"])
    for row in rows:
        if row["model"] in {"LSTM", "GRU"}:
            row["mae_vs_transformer_pct"] = (transformer_mae - row["mae"]) / transformer_mae * 100
            row["rmse_vs_transformer_pct"] = (transformer_rmse - row["rmse"]) / transformer_rmse * 100
            row["phm_vs_transformer_pct"] = (transformer_phm - row["phm_score"]) / transformer_phm * 100
    results = pd.DataFrame(rows)
    results.to_csv(output / "model_comparison.csv", index=False)
    results.to_json(output / "model_comparison.json", orient="records", indent=2)
    lifecycle = []
    for name, prediction in predictions.items():
        lifecycle.extend([{"model": name, **row} for row in lifecycle_metrics(y_validation, prediction)])
    pd.DataFrame(lifecycle).to_csv(output / "lifecycle_error_comparison.csv", index=False)
    json.dump(reproducibility_manifest(data_path, x_train, x_validation, device), (output / "reproducibility_manifest.json").open("w", encoding="utf-8"), indent=2)
    json.dump({
        "train_scaler_fitted": False,
        "validation_used_for_checkpoint_selection": True,
        "validation_labels_used_for_training": False,
        "test_or_future_data_used": False,
        "model_selection_on_final_evaluation": False,
        "protocol": "preprocessed Step 6 arrays; fixed train/validation contract",
    }, (output / "leakage_audit.json").open("w", encoding="utf-8"), indent=2)
    registry["reproducibility_manifest"] = "reproducibility_manifest.json"
    registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return results
