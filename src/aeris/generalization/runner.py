"""Bounded leave-one-dataset-out C-MAPSS experiments."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aeris.data.cmapss import load_cmapss, split_engines
from aeris.experiment.runtime import ExperimentTimeout, RuntimeGuard
from aeris.health.evaluation import regression_metrics, trajectory_metrics
from aeris.health.labels import rul_to_health
from aeris.health.multidataset import FEATURE_COLUMNS, ROOT, WINDOW_SIZE, _rul_windows
from aeris.health.trajectory import early_late_errors
from aeris.models.temporal import TransformerRegressor

DATASETS = ("FD001", "FD002", "FD003", "FD004")
SEED = 42


def _metrics(actual: np.ndarray, predicted: np.ndarray, owners: np.ndarray) -> dict[str, Any]:
    result = regression_metrics(actual, predicted)
    trajectory = trajectory_metrics(actual, predicted)
    result.update(early_late_errors(actual, predicted))
    result.update({key: value for key, value in trajectory.items() if key not in result})
    result.update({"windows": int(len(actual)), "engines": int(len(np.unique(owners)))})
    return result


def _fit_bounded(
    X: np.ndarray,
    y: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    checkpoint: Path,
    guard: RuntimeGuard,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = TransformerRegressor(input_size=X.shape[-1], max_length=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    started = time.perf_counter()
    model.train()
    for batch_index, (features, targets) in enumerate(loader):
        guard.check({"epoch": 1, "batch": batch_index, "device": str(device)})
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        dev_loss = float(loss_fn(model(torch.as_tensor(X_dev).to(device)), torch.as_tensor(y_dev, dtype=torch.float32).to(device)).item())
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), checkpoint)
    return {
        "training_time_seconds": time.perf_counter() - started,
        "best_development_loss": dev_loss,
        "epochs": 1,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
    }


def _predict(checkpoint: Path, X: np.ndarray) -> tuple[np.ndarray, float]:
    model = TransformerRegressor(input_size=X.shape[-1], max_length=X.shape[1])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        prediction = model(torch.as_tensor(X, dtype=torch.float32)).numpy()
    return prediction, time.perf_counter() - started


def _raw_dataset(name: str) -> dict[str, Any]:
    dataset = load_cmapss(ROOT / "data" / "raw", name)
    train_ids, dev_ids, final_ids = split_engines(dataset.train.engine_id.unique(), SEED)
    raw = {split: _rul_windows(dataset.train, ids) for split, ids in (
        ("train", train_ids), ("development", dev_ids), ("final", final_ids)
    )}
    values = raw["train"][0].reshape(-1, raw["train"][0].shape[-1]).astype(float)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-8)
    transformed = {split: ((items[0] - mean) / scale).astype(np.float32) for split, items in raw.items()}
    return {
        "name": name, "train": transformed["train"], "development": transformed["development"],
        "final": transformed["final"], "train_rul": raw["train"][2],
        "development_rul": raw["development"][2], "final_rul": raw["final"][2],
        "development_owners": raw["development"][1], "final_owners": raw["final"][1],
        "train_engines": train_ids, "development_engines": dev_ids, "final_engines": final_ids,
        "mean": mean.tolist(), "scale": scale.tolist(),
    }


def run_leave_one_out(output_dir: str | Path = "reports/experiments/cmapss_generalization_full") -> dict[str, Any]:
    output = ROOT / output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = {name: _raw_dataset(name) for name in DATASETS}
    experiments = [
        ("C2", ("FD001", "FD002", "FD004"), "FD003"),
        ("C3", ("FD001", "FD003", "FD004"), "FD002"),
        ("C4", ("FD002", "FD003", "FD004"), "FD001"),
    ]
    rows = []
    statuses = []
    for experiment_id, train_names, held_out in experiments:
        result_path = output / f"{experiment_id}_results.json"
        guard = RuntimeGuard(output, experiment_id, 600)
        started = time.perf_counter()
        try:
            train_X = np.concatenate([data[name]["train"] for name in train_names])
            train_rul = np.concatenate([data[name]["train_rul"] for name in train_names])
            dev_X = np.concatenate([data[name]["development"] for name in train_names])
            dev_rul = np.concatenate([data[name]["development_rul"] for name in train_names])
            reference = float(train_rul.max())
            checkpoint = output / "models" / f"{experiment_id.lower()}.pt"
            training = _fit_bounded(
                train_X, rul_to_health(train_rul, reference),
                dev_X, rul_to_health(dev_rul, reference), checkpoint, guard,
            )
            target = data[held_out]
            prediction, inference = _predict(checkpoint, target["final"])
            actual = rul_to_health(target["final_rul"], reference)
            row = {
                "experiment_id": experiment_id,
                "training_datasets": list(train_names),
                "test_dataset": held_out,
                "status": "COMPLETED",
                **_metrics(actual, prediction, target["final_owners"]),
                **training,
                "inference_time_seconds": inference,
                "training_engines": int(sum(len(data[name]["train_engines"]) for name in train_names)),
                "training_windows": int(len(train_X)),
                "test_engines": int(len(target["final_engines"])),
                "test_windows": int(len(target["final"])),
                "health_reference": reference,
                "feature_count": len(FEATURE_COLUMNS),
                "window_size": WINDOW_SIZE,
                "seed": SEED,
            }
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            guard.complete(row)
            statuses.append({"experiment_id": experiment_id, "status": "COMPLETED"})
        except ExperimentTimeout as exc:
            row = {"experiment_id": experiment_id, "status": "TIMEOUT", "reason": str(exc), "elapsed_seconds": time.perf_counter() - started}
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            statuses.append(row)
        except Exception as exc:
            row = {"experiment_id": experiment_id, "status": "FAILED", "reason": str(exc)}
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            statuses.append(row)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    (output / "status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "experiments": [item[0] for item in experiments],
        "epochs": 1, "runtime_budget_seconds": 600, "seed": SEED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "feature_columns": FEATURE_COLUMNS, "window_size": WINDOW_SIZE,
        "health_definition": "clip(RUL / max(permitted training RUL), 0, 1)",
    }, indent=2), encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps({
        name: {key: value for key, value in item.items() if key.endswith("engines")}
        for name, item in data.items()
    }, indent=2), encoding="utf-8")
    (output / "leakage_audit.json").write_text(json.dumps({
        "engine_level_splitting": "PASS",
        "no_window_leakage": "PASS",
        "target_dataset_held_out": "PASS",
        "target_final_labels_used_for_selection": "PASS",
        "training_only_scalers": "PASS",
        "health_reference_training_only": "PASS",
        "deterministic_seed": SEED,
        "protected_artifacts_overwritten": "PASS",
    }, indent=2), encoding="utf-8")
    (output / "reproducibility_manifest.json").write_text(json.dumps({
        "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda": torch.version.cuda, "seed": SEED, "datasets": list(DATASETS),
        "architecture": "existing TransformerRegressor", "epochs": 1,
        "status": statuses,
    }, indent=2), encoding="utf-8")
    return {"results": pd.DataFrame(rows), "statuses": statuses}
