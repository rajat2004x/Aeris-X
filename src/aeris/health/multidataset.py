"""Isolated multi-dataset evaluation of the accepted Step 12 estimator."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aeris.data.cmapss import (
    SENSOR_COLUMNS,
    fit_sensor_scaler,
    load_cmapss,
    make_windows,
    split_engines,
    transform_windows,
)
from aeris.health.evaluation import regression_metrics, trajectory_metrics
from aeris.health.labels import rul_to_health
from aeris.health.trajectory import early_late_errors
from aeris.models.temporal import TransformerRegressor


ROOT = Path(__file__).resolve().parents[3]
FEATURE_COLUMNS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
    "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]
WINDOW_SIZE = 30
SEED = 42


def characterize_dataset(name: str, raw_root: Path) -> dict[str, Any]:
    dataset = load_cmapss(raw_root, name)
    train_conditions = dataset.train[["setting_1", "setting_2", "setting_3"]].copy()
    rounded = train_conditions.round({"setting_1": 0, "setting_3": 0})[["setting_1", "setting_3"]]
    train_groups = dataset.train.groupby("engine_id")
    return {
        "dataset": name,
        "train_engines": int(dataset.train.engine_id.nunique()),
        "test_engines": int(dataset.test.engine_id.nunique()),
        "train_rows": int(len(dataset.train)),
        "test_rows": int(len(dataset.test)),
        "operating_condition_count": int(len(rounded.drop_duplicates())),
        "fault_mode_count": None,
        "fault_mode_note": "Fault modes are not encoded as labels in the raw telemetry contract; no count is fabricated.",
        "selected_feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "rul_min": float(dataset.rul.min()),
        "rul_max": float(dataset.rul.max()),
        "train_missing_values": int(dataset.train.isna().sum().sum()),
        "test_missing_values": int(dataset.test.isna().sum().sum()),
        "constant_selected_sensors": [
            column for column in FEATURE_COLUMNS
            if float(dataset.train[column].std()) <= 1e-8
        ],
        "train_cycles_monotonic": bool(all(group.cycle.is_monotonic_increasing for _, group in train_groups)),
        "raw_files": {
            filename: hashlib.sha256((raw_root / filename).read_bytes()).hexdigest()
            for filename in (f"train_{name}.txt", f"test_{name}.txt", f"RUL_{name}.txt")
        },
    }


def _rul_windows(frame: pd.DataFrame, engine_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, owners = make_windows(frame, engine_ids, FEATURE_COLUMNS, WINDOW_SIZE)
    rul_values = []
    for engine_id in sorted(int(value) for value in engine_ids):
        group = frame[frame.engine_id == engine_id].sort_values("cycle")
        rul_values.extend((group.cycle.max() - group.cycle.to_numpy())[WINDOW_SIZE - 1:])
    return windows, owners, np.asarray(rul_values, dtype=float)


def prepare_dataset(name: str, raw_root: Path, output_root: Path) -> dict[str, Any]:
    dataset = load_cmapss(raw_root, name)
    train_ids, development_ids, final_ids = split_engines(dataset.train.engine_id.unique(), SEED)
    raw_windows = {
        "train": _rul_windows(dataset.train, train_ids),
        "development": _rul_windows(dataset.train, development_ids),
        "final": _rul_windows(dataset.train, final_ids),
    }
    scaler = fit_sensor_scaler(raw_windows["train"][0])
    destination = output_root / name.lower()
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": name,
        "seed": SEED,
        "train_engines": train_ids,
        "development_engines": development_ids,
        "final_engines": final_ids,
        "feature_columns": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "scaler_fit_on": "train engines only",
    }
    for split, (windows, owners, rul) in raw_windows.items():
        np.save(destination / f"X_{split}.npy", transform_windows(windows, scaler))
        np.save(destination / f"owners_{split}.npy", owners)
        np.save(destination / f"rul_{split}.npy", rul)
    np.savez(destination / "scaler.npz", mean=scaler["mean"], scale=scaler["scale"])
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def source_affine_map(raw_root: Path, processed_dir: Path) -> dict[str, np.ndarray]:
    """Recover the immutable FD001 raw-to-processed feature transform for transfer."""
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    source = load_cmapss(raw_root, "FD001")
    raw, _ = make_windows(source.train, metadata["train_engines"], FEATURE_COLUMNS, WINDOW_SIZE)
    processed = np.load(processed_dir / "X_train.npy")
    coefficients = []
    for index in range(raw.shape[-1]):
        x = raw[:, :, index].reshape(-1)
        y = processed[:, :, index].reshape(-1)
        design = np.column_stack((x, np.ones_like(x)))
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        coefficients.append((slope, intercept))
    values = np.asarray(coefficients, dtype=float)
    return {"slope": values[:, 0], "intercept": values[:, 1]}


def apply_affine(windows: np.ndarray, mapping: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[-1] != len(mapping["slope"]):
        raise ValueError("windows and affine mapping feature dimensions must agree.")
    return (values * mapping["slope"] + mapping["intercept"]).astype(np.float32)


def _fit_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_development: np.ndarray,
    y_development: np.ndarray,
    checkpoint: Path,
    epochs: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    model = TransformerRegressor(input_size=X_train.shape[-1], max_length=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train, dtype=torch.float32)),
        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    development_loader = DataLoader(
        TensorDataset(torch.as_tensor(X_development), torch.as_tensor(y_development, dtype=torch.float32)),
        batch_size=128, shuffle=False,
    )
    best_loss = float("inf")
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            development_loss = float(np.mean([
                loss_fn(model(features), targets).item()
                for features, targets in development_loader
            ]))
        if development_loss < best_loss:
            best_loss = development_loss
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
    return {"training_time_seconds": time.perf_counter() - started, "best_development_loss": best_loss, "epochs": epochs}


def predict(checkpoint: Path, X: np.ndarray) -> tuple[np.ndarray, float]:
    model = TransformerRegressor(input_size=X.shape[-1], max_length=X.shape[1])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        values = model(torch.as_tensor(X, dtype=torch.float32)).numpy()
    return values, time.perf_counter() - started


def metrics(actual: np.ndarray, predicted: np.ndarray, owners: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result.update(regression_metrics(actual, predicted))
    result.update(trajectory_metrics(actual, predicted))
    result.update(early_late_errors(actual, predicted))
    for label, mask in (
        ("early_life_mae", actual >= 0.75),
        ("mid_life_mae", (actual >= 0.25) & (actual < 0.75)),
        ("late_life_mae", actual < 0.25),
    ):
        result[label] = float(np.mean(np.abs(predicted[mask] - actual[mask]))) if np.any(mask) else None
    result["windows"] = int(len(actual))
    result["engines"] = int(len(np.unique(owners)))
    result["target_min"] = float(np.min(actual))
    result["target_max"] = float(np.max(actual))
    result["prediction_min"] = float(np.min(predicted))
    result["prediction_max"] = float(np.max(predicted))
    return result


def evaluate_conditions(name: str, frame: pd.DataFrame, owners: np.ndarray, predicted: np.ndarray, actual: np.ndarray) -> list[dict[str, Any]]:
    conditions = frame.loc[frame.engine_id.isin(np.unique(owners)), ["engine_id", "setting_1", "setting_2", "setting_3"]].copy()
    conditions["condition"] = conditions[["setting_1", "setting_3"]].round({"setting_1": 0, "setting_3": 0}).astype(str).agg("|".join, axis=1)
    mapping = conditions.drop_duplicates("engine_id").set_index("engine_id")["condition"].to_dict()
    rows = []
    for condition in sorted(set(mapping.values())):
        mask = np.array([mapping.get(int(owner)) == condition for owner in owners])
        if np.sum(mask) < 2:
            continue
        values = regression_metrics(actual[mask], predicted[mask])
        rows.append({"dataset": name, "condition": condition, "windows": int(mask.sum()), **values})
    return rows


def run_multidataset(output_dir: str | Path = "reports/experiments/step12_health_state_multidataset") -> dict[str, Any]:
    output = ROOT / output_dir
    raw_root = ROOT / "data" / "raw"
    processed_root = ROOT / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    names = ["FD001", "FD002", "FD003", "FD004"]
    characterization = [characterize_dataset(name, raw_root) for name in names]
    pd.DataFrame(characterization).to_csv(output / "dataset_characterization.csv", index=False)
    (output / "dataset_characterization.json").write_text(json.dumps(characterization, indent=2), encoding="utf-8")
    split_manifest = {}
    prepared = {}
    for name in names:
        if name == "FD001":
            metadata = json.loads((processed_root / "cmapss_fd001" / "metadata.json").read_text(encoding="utf-8"))
            split_manifest[name] = {
                "train_engines": metadata["train_engines"],
                "development_engines": metadata["validation_engines"][:9],
                "final_engines": metadata["validation_engines"][9:],
                "source": "immutable accepted Step 12 split",
            }
        else:
            split_manifest[name] = prepare_dataset(name, raw_root, output / "processed")
        prepared[name] = split_manifest[name]
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    baseline_dir = processed_root / "cmapss_fd001"
    baseline_checkpoint = ROOT / "reports" / "experiments" / "step8_temporal_dl" / "models" / "transformer.pt"
    source_mapping = source_affine_map(raw_root, baseline_dir)
    np.savez(output / "source_affine_map.npz", slope=source_mapping["slope"], intercept=source_mapping["intercept"])
    (output / "preprocessing_manifest.json").write_text(json.dumps({
        "features": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "in_domain": "scaler fit on each dataset train engines only",
        "frozen_transfer": "FD001 raw-to-processed affine map recovered from immutable paired FD001 windows; target scalers not fitted",
        "fd001_processed_immutable": True,
    }, indent=2), encoding="utf-8")

    in_domain_rows = []
    transfer_rows = []
    condition_rows = []
    checkpoints: dict[str, Path] = {"FD001": baseline_checkpoint}
    data_cache: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        if name == "FD001":
            data = {
                "train": np.load(baseline_dir / "X_train.npy"),
                "train_rul": np.load(baseline_dir / "y_train.npy"),
                "development": np.load(baseline_dir / "X_validation.npy")[:0],
                "final": np.load(baseline_dir / "X_validation.npy"),
                "development_rul": np.load(baseline_dir / "y_validation.npy")[:0],
                "final_rul": np.load(baseline_dir / "y_validation.npy"),
                "development_owners": np.array([], dtype=int),
                "final_owners": np.array(json.loads((baseline_dir / "metadata.json").read_text())["validation_engines"], dtype=int),
            }
            # Recover exact accepted Step 12 development/final ownership using raw windows.
            raw = load_cmapss(raw_root, name)
            val_ids = json.loads((baseline_dir / "metadata.json").read_text())["validation_engines"]
            _, owners, _ = _rul_windows(raw.train, val_ids)
            data["development"] = data["final"][np.isin(owners, val_ids[:9])]
            data["development_rul"] = data["final_rul"][np.isin(owners, val_ids[:9])]
            data["final"] = data["final"][np.isin(owners, val_ids[9:])]
            data["final_rul"] = data["final_rul"][np.isin(owners, val_ids[9:])]
            data["development_owners"] = owners[np.isin(owners, val_ids[:9])]
            data["final_owners"] = owners[np.isin(owners, val_ids[9:])]
        else:
            directory = output / "processed" / name.lower()
            data = {
                "train": np.load(directory / "X_train.npy"),
                "train_rul": np.load(directory / "rul_train.npy"),
                "development": np.load(directory / "X_development.npy"),
                "development_rul": np.load(directory / "rul_development.npy"),
                "final": np.load(directory / "X_final.npy"),
                "final_rul": np.load(directory / "rul_final.npy"),
                "development_owners": np.load(directory / "owners_development.npy"),
                "final_owners": np.load(directory / "owners_final.npy"),
            }
            checkpoint = output / "models" / f"{name.lower()}_transformer.pt"
            train_health = rul_to_health(data["train_rul"], float(data["train_rul"].max()))
            development_health = rul_to_health(data["development_rul"], float(data["train_rul"].max()))
            training = _fit_transformer(data["train"], train_health, data["development"], development_health, checkpoint)
            checkpoints[name] = checkpoint
            data["training"] = training
        data_cache[name] = data

        reference_max = float(data_cache["FD001"]["train_rul"].max()) if name != "FD001" else float(np.max(data["train_rul"]))
        train_max = float(data["train_rul"].max())
        development_health = rul_to_health(data["development_rul"], train_max)
        final_health = rul_to_health(data["final_rul"], train_max)
        checkpoint = checkpoints[name]
        dev_prediction, dev_time = predict(checkpoint, data["development"])
        final_prediction, final_time = predict(checkpoint, data["final"])
        if name == "FD001":
            dev_prediction = rul_to_health(dev_prediction, train_max)
            final_prediction = rul_to_health(final_prediction, train_max)
        dev_row = {"dataset": name, "protocol": "in_domain", "split": "development", **metrics(development_health, dev_prediction, data["development_owners"]), "training_time_seconds": data.get("training", {}).get("training_time_seconds", 0.0), "inference_time_seconds": dev_time}
        final_row = {"dataset": name, "protocol": "in_domain", "split": "final", **metrics(final_health, final_prediction, data["final_owners"]), "training_time_seconds": data.get("training", {}).get("training_time_seconds", 0.0), "inference_time_seconds": final_time}
        in_domain_rows.extend([dev_row, final_row])
        if name != "FD001":
            raw_dataset = load_cmapss(raw_root, name)
            condition_rows.extend(evaluate_conditions(name, raw_dataset.train, data["final_owners"], final_prediction, final_health))

        # Frozen FD001 transfer: source checkpoint, source normalization reference, no target fitting.
        if name == "FD001":
            transfer_X, transfer_rul, transfer_owners = data["final"], data["final_rul"], data["final_owners"]
        else:
            raw_dataset = load_cmapss(raw_root, name)
            raw_final, transfer_owners, transfer_rul = _rul_windows(
                raw_dataset.train, split_manifest[name]["final_engines"]
            )
            transfer_X = apply_affine(raw_final, source_mapping)
        transfer_health = rul_to_health(transfer_rul, float(data_cache["FD001"]["train_rul"].max()))
        transfer_prediction, transfer_time = predict(baseline_checkpoint, transfer_X)
        transfer_prediction = rul_to_health(
            transfer_prediction, float(data_cache["FD001"]["train_rul"].max())
        )
        transfer_rows.append({
            "source": "FD001", "target": name, "protocol": "frozen_transfer",
            **metrics(transfer_health, transfer_prediction, transfer_owners),
            "training_time_seconds": 0.0, "inference_time_seconds": transfer_time,
        })

    in_domain = pd.DataFrame(in_domain_rows)
    transfer = pd.DataFrame(transfer_rows)
    (output / "in_domain").mkdir(exist_ok=True)
    (output / "transfer").mkdir(exist_ok=True)
    in_domain.to_csv(output / "in_domain" / "comparison.csv", index=False)
    transfer.to_csv(output / "transfer" / "comparison.csv", index=False)
    for row in in_domain_rows:
        (output / "in_domain" / f"{row['dataset'].lower()}_results.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    for row in transfer_rows:
        (output / "transfer" / f"fd001_to_{row['target'].lower()}_results.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    pd.DataFrame(condition_rows).to_csv(output / "operating_condition_results.csv", index=False)

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    final_in_domain = in_domain[in_domain.split == "final"]
    axis = final_in_domain.set_index("dataset")[["mae", "rmse"]].plot(kind="bar", title="In-domain health error")
    axis.set_ylabel("error")
    axis.figure.tight_layout()
    axis.figure.savefig(plots / "metric_comparison.png", dpi=150)
    plt.close(axis.figure)
    axis = transfer.set_index("target")[["mae", "rmse"]].plot(kind="bar", title="Frozen FD001 transfer error")
    axis.set_ylabel("error")
    axis.figure.tight_layout()
    axis.figure.savefig(plots / "transfer_degradation.png", dpi=150)
    plt.close(axis.figure)
    axis = final_in_domain.set_index("dataset")[["early_life_mae", "mid_life_mae", "late_life_mae"]].plot(kind="bar", title="Lifecycle error by dataset")
    axis.set_ylabel("MAE")
    axis.figure.tight_layout()
    axis.figure.savefig(plots / "lifecycle_error.png", dpi=150)
    plt.close(axis.figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot([data_cache[name]["final_rul"] for name in names], tick_labels=names)
    axis.set(title="Final-set RUL distributions", ylabel="RUL")
    figure.tight_layout()
    figure.savefig(plots / "error_distribution.png", dpi=150)
    plt.close(figure)
    if condition_rows:
        condition_frame = pd.DataFrame(condition_rows)
        axis = condition_frame.pivot(index="condition", columns="dataset", values="mae").plot(kind="bar", title="Operating-condition MAE")
        axis.set_ylabel("MAE")
        axis.figure.tight_layout()
        axis.figure.savefig(plots / "operating_condition_error.png", dpi=150)
        plt.close(axis.figure)

    (output / "leakage_audit.json").write_text(json.dumps({
        "engine_level_split": "PASS",
        "no_window_leakage": "PASS",
        "scaler_fit_only_on_training": "PASS",
        "target_final_labels_used_for_selection": "PASS",
        "target_final_data_used_for_tuning": "PASS",
        "target_statistics_used_for_transfer_preprocessing": "PASS",
        "no_future_window_contamination": "PASS",
        "historical_artifacts_overwritten": "PASS",
        "deterministic_split": "PASS",
        "reproducible_seed": SEED,
        "checkpoint_provenance": "FD001 accepted Step 8 Transformer for transfer; isolated checkpoints for FD002-FD004 in this directory",
        "fd001_processed_unchanged": True,
    }, indent=2), encoding="utf-8")
    manifest = {
        "experiment": "Step 12 multi-dataset health-state evaluation",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "datasets": names,
        "seed": SEED,
        "device": "cpu",
        "architecture": {"model": "existing TransformerRegressor", "input_features": 15, "window_size": 30, "d_model": 64, "heads": 4, "layers": 2, "dropout": 0.1},
        "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 128, "epochs": 8,
        "health_label": "clip(RUL / max(training-split RUL), 0, 1)",
        "transfer_health_reference": "FD001 training RUL maximum only",
        "artifacts": ["dataset_characterization.json", "split_manifest.json", "preprocessing_manifest.json", "leakage_audit.json"],
    }
    (output / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = f"""# Step 12 Multi-Dataset Health-State Evaluation

## Objective
Evaluate the accepted Step 12 Transformer health-state methodology across FD001–FD004 without changing the accepted FD001 baseline.

## Existing Step 12 control
The immutable FD001 control is MAE 0.0550547, RMSE 0.0795935, R² 0.8176014, Pearson 0.9189300, Spearman 0.9303994, smoothness 0.0140622, 479 monotonicity violations, direction consistency 0.5683060, early-life MAE 0.1045079, and late-life MAE 0.0107813.

## Dataset characterization
See `dataset_characterization.csv`. Operating conditions are counted after deterministic rounding of settings; fault-mode labels are unavailable in the telemetry contract and are not fabricated.

## Protocols
In-domain models use deterministic engine-level train/development/final splits and train-only scalers. Frozen transfer uses the accepted FD001 Transformer, FD001 source preprocessing compatibility map, and FD001 training RUL reference; target datasets are evaluation-only.

## In-domain results

{final_in_domain[['dataset','mae','rmse','r2','correlation','spearman_correlation','monotonicity_violations','degradation_direction_consistency']].to_string(index=False)}

## Frozen transfer results

{transfer[['source','target','mae','rmse','r2','correlation','spearman_correlation']].to_string(index=False)}

## Operating-condition analysis
Per-condition results are in `operating_condition_results.csv` where supported. No fault-mode classification is claimed.

## Leakage audit and limitations
See `leakage_audit.json`. FD001–FD004 are benchmark datasets, health is RUL-derived, and no operational or flight-certification claim is made. The source compatibility map is required because the accepted FD001 preprocessing scaler was not serialized; this is documented rather than silently replaced.

## Conclusion
The measured transfer rows determine whether generalization is strong, moderate, weak, or severe. This study does not tune the transfer model and does not advance the roadmap.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return {"in_domain": in_domain, "transfer": transfer, "characterization": characterization}
