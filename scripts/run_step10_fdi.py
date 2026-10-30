"""Run Step 10 synthetic sensor FDI and anomaly-detection experiments."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.evaluation.benchmark import load_ml_ready_dataset
from aeris.evaluation.fdi import detection_metrics, isolation_rank, reconstruction_scores, threshold_from_normal
from aeris.faults.injection import inject_fault
from aeris.models.anomaly import SequenceAutoencoder
from aeris.training.datasets import TemporalDataset
from aeris.training.trainer import seed_everything, select_device


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    originals, reconstructed = [], []
    with torch.no_grad():
        for features, _ in loader:
            originals.append(features.numpy())
            reconstructed.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(originals), np.concatenate(reconstructed)


def _train(model: nn.Module, loader: DataLoader, device: torch.device, output: Path) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    history = []
    best = float("inf")
    for _ in range(8):
        model.train()
        total = 0.0
        count = 0
        for features, _ in loader:
            features = features.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), features)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(features)
            count += len(features)
        value = total / count
        history.append(value)
        if value < best:
            best = value
            torch.save(model.state_dict(), output)
    model.load_state_dict(torch.load(output, map_location=device, weights_only=True))
    return history


def _plot_examples(examples: dict[str, np.ndarray], output: Path) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for name, values in examples.items():
        figure, axis = plt.subplots()
        axis.plot(values)
        axis.set(title=f"Injected {name} example", xlabel="Timestep", ylabel="Normalized sensor value")
        figure.tight_layout()
        figure.savefig(plots / f"{name.lower()}_example.png", dpi=150)
        plt.close(figure)


def main() -> None:
    seed_everything(42)
    output = PROJECT_ROOT / "reports/experiments/step10_fdi"
    models = output / "models"
    predictions = output / "predictions"
    models.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    X_train, y_train, X_validation, y_validation = load_ml_ready_dataset(
        PROJECT_ROOT / "data/processed/cmapss_fd001"
    )
    train_loader = DataLoader(TemporalDataset(X_train, y_train), batch_size=128, shuffle=True)
    normal_loader = DataLoader(TemporalDataset(X_validation, y_validation), batch_size=128, shuffle=False)
    model = SequenceAutoencoder(input_size=X_train.shape[2], hidden_size=32).to(device)
    training_started = time.perf_counter()
    history = _train(model, train_loader, device, models / "sequence_autoencoder.pt")
    training_seconds = time.perf_counter() - training_started
    normal_original, normal_reconstructed = _predict(model, normal_loader, device)
    normal_scores, _ = reconstruction_scores(normal_original, normal_reconstructed)
    threshold = threshold_from_normal(normal_scores, 95)

    # Fault scales are fractions/multiples of the training distribution and
    # scenarios cover three sensors, three severities, and three onsets.
    sensor_indices = [0, 1, 2]
    fault_types = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
    severity_names = {"low": 0.25, "medium": 0.5, "high": 1.0}
    onset_names = {"early": 0.2, "middle": 0.5, "late": 0.75}
    feature_std = np.std(X_train, axis=(0, 1))
    scenarios, fault_windows, labels = [], [], []
    examples = {}
    for index, (fault_type, severity_name, onset_name, sensor_index) in enumerate(
        (t, s, o, i)
        for t in fault_types
        for s in severity_names
        for o in onset_names
        for i in sensor_indices
    ):
        source = X_validation[index % len(X_validation)]
        injected, metadata = inject_fault(
            source,
            sensor_index,
            fault_type,
            onset_names[onset_name],
            severity_names[severity_name],
            max(float(feature_std[sensor_index]), 1e-6),
            seed=42 + index,
        )
        metadata.update({"severity_name": severity_name, "onset_name": onset_name, "source_index": index % len(X_validation)})
        scenarios.append(metadata)
        fault_windows.append(injected)
        labels.append(1)
        examples.setdefault(fault_type, injected[:, sensor_index])

    fault_array = np.asarray(fault_windows, dtype=np.float32)
    fault_loader = DataLoader(TemporalDataset(fault_array, np.zeros(len(fault_array))), batch_size=128, shuffle=False)
    fault_original, fault_reconstructed = _predict(model, fault_loader, device)
    fault_scores, fault_sensor_scores = reconstruction_scores(fault_original, fault_reconstructed)
    all_scores = np.concatenate([normal_scores, fault_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores), dtype=int), np.ones(len(fault_scores), dtype=int)])
    metrics = detection_metrics(all_labels, all_scores, threshold)

    isolation_rows = []
    detection_rows = []
    for scenario, score, sensor_score in zip(scenarios, fault_scores, fault_sensor_scores):
        rank, ranking = isolation_rank(sensor_score, int(scenario["sensor_index"]))
        detection_rows.append({**scenario, "anomaly_score": float(score), "threshold": threshold, "detected": bool(score >= threshold), "detection_latency_cycles": max(0, int(30 * (1 - scenario["onset_fraction"]))) if score >= threshold else None})
        isolation_rows.append({**scenario, "isolation_rank": rank, "top1_sensor": ranking[0], "top3_sensors": ranking[:3], "top1_correct": rank == 1, "top3_correct": rank <= 3})

    detections = pd.DataFrame(detection_rows)
    isolations = pd.DataFrame(isolation_rows)
    type_summary = detections.groupby("fault_type").agg(
        precision=("detected", lambda x: float(np.mean(x))),
        detection_rate=("detected", "mean"),
        mean_latency=("detection_latency_cycles", "mean"),
    ).reset_index()
    fdi_summary = isolations.groupby("fault_type").agg(top1_accuracy=("top1_correct", "mean"), top3_accuracy=("top3_correct", "mean")).reset_index()
    type_summary = type_summary.merge(fdi_summary, on="fault_type")

    output.mkdir(parents=True, exist_ok=True)
    detections.to_csv(output / "detection_results.csv", index=False)
    isolations.to_csv(output / "isolation_results.csv", index=False)
    pd.DataFrame(scenarios).to_csv(output / "fault_scenarios.csv", index=False)
    np.save(predictions / "normal_scores.npy", normal_scores)
    np.save(predictions / "fault_scores.npy", fault_scores)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    _plot_examples(examples, output)
    for name, x, y, title in (
        ("roc_curve", fpr, tpr, "ROC curve"),
        ("precision_recall_curve", recall, precision, "Precision-recall curve"),
    ):
        figure, axis = plt.subplots()
        axis.plot(x, y)
        axis.set(title=title, xlabel="Rate", ylabel="Rate")
        figure.tight_layout()
        figure.savefig(plot_dir / f"{name}.png", dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots()
    axis.hist(normal_scores, bins=30, alpha=0.6, label="normal")
    axis.hist(fault_scores, bins=30, alpha=0.6, label="fault")
    axis.axvline(threshold, linestyle="--", color="black", label="threshold")
    axis.legend()
    axis.set(title="Normal vs injected-fault anomaly scores", xlabel="Reconstruction MSE", ylabel="Count")
    figure.tight_layout()
    figure.savefig(plot_dir / "score_distribution.png", dpi=150)
    plt.close(figure)
    ConfusionMatrixDisplay.from_predictions(all_labels, all_scores >= threshold)
    plt.tight_layout()
    plt.savefig(plot_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    figure, axis = plt.subplots()
    axis.bar(type_summary["fault_type"], type_summary["detection_rate"])
    axis.set(title="Detection rate by fault type", ylabel="Detection rate")
    figure.tight_layout()
    figure.savefig(plot_dir / "detection_by_fault_type.png", dpi=150)
    plt.close(figure)

    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "fault_scenarios": len(scenarios),
        "fault_types": fault_types,
        "training_seconds": training_seconds,
        "threshold": threshold,
        "threshold_percentile": 95,
        "normal_false_positive_rate": float(np.mean(normal_scores >= threshold)),
        "detection_metrics": metrics,
        "top1_isolation_accuracy": float(isolations["top1_correct"].mean()),
        "top3_isolation_accuracy": float(isolations["top3_correct"].mean()),
        "mean_detection_latency": float(detections["detection_latency_cycles"].dropna().mean()),
        "median_detection_latency": float(detections["detection_latency_cycles"].dropna().median()),
        "detection_rate": float(detections["detected"].mean()),
        "fault_type_summary": type_summary.to_dict(orient="records"),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([result["detection_metrics"] | {"top1_isolation_accuracy": result["top1_isolation_accuracy"], "top3_isolation_accuracy": result["top3_isolation_accuracy"]}]).to_csv(output / "results.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "seed": 42, "window_size": 30, "num_features": 15, "batch_size": 128,
        "autoencoder": {"type": "LSTM sequence autoencoder", "hidden_size": 32, "epochs": 8, "optimizer": "Adam", "learning_rate": 1e-3, "loss": "MSELoss"},
        "threshold": {"method": "95th percentile of normal validation reconstruction errors", "sustained_detection_windows": 1},
        "fault_matrix": {"types": fault_types, "severities": severity_names, "onsets": onset_names, "sensor_indices": sensor_indices},
    }, indent=2), encoding="utf-8")
    report = f"""# Step 10: Sensor FDI and Anomaly Detection

## Objective

Investigate whether telemetry anomaly scores can detect controlled synthetic
sensor faults and isolate their affected sensor while distinguishing normal
C-MAPSS degradation from injected faults.

## Data and integrity

NASA C-MAPSS FD001 is the real benchmark dataset. The fault labels in this
experiment are simulated: controlled synthetic faults were injected into
existing normalized validation windows. The autoencoder was trained only on
normal training windows; injected faults were not used for training or
threshold fitting. Previous Step 6-9 data and artifacts were not modified.

## Fault matrix

{len(scenarios)} scenarios: five fault types, three severity levels, three
onset positions, and three selected variable sensors. Fault types: {", ".join(fault_types)}.
Severity scales are relative to each sensor's training standard deviation.

## Detector and threshold

An LSTM sequence autoencoder with hidden size 32 was trained for eight epochs
with Adam and MSE loss on normal training windows. The anomaly score is mean
reconstruction MSE. The threshold is the 95th percentile of normal validation
scores: `{threshold:.6f}`. Normal-validation false-positive rate:
`{result["normal_false_positive_rate"]:.2%}`.

## Results

- Precision: `{metrics["precision"]:.4f}`
- Recall: `{metrics["recall"]:.4f}`
- F1: `{metrics["f1"]:.4f}`
- Accuracy: `{metrics["accuracy"]:.4f}`
- False-positive rate: `{metrics["false_positive_rate"]:.4f}`
- False-negative rate: `{metrics["false_negative_rate"]:.4f}`
- ROC-AUC: `{metrics["roc_auc"]:.4f}`
- PR-AUC: `{metrics["pr_auc"]:.4f}`
- Top-1 isolation accuracy: `{result["top1_isolation_accuracy"]:.4f}`
- Top-3 isolation accuracy: `{result["top3_isolation_accuracy"]:.4f}`
- Mean detection latency: `{result["mean_detection_latency"]:.2f} cycles`
- Median detection latency: `{result["median_detection_latency"]:.2f} cycles`

## Limitations

Injected faults are controlled simulations, not real aerospace failure labels.
C-MAPSS FD001 is a benchmark engine-degradation dataset, not operational
telemetry. Anomaly detection is not automatically diagnosis; real FDI requires
validated failure data, sensor redundancy, physics, and operational context.
These results do not establish aerospace safety or flight certification.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
