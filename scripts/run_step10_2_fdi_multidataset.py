"""Execute the isolated, leakage-audited Step 10.2 FDI experiment."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.data.adapters.cmapss_adapter import load_prepared_cmapss
from aeris.evaluation.anomaly import fit_robust_baseline, robust_sensor_scores
from aeris.evaluation.fdi import reconstruction_scores
from aeris.faults.injection import inject_fault
from aeris.fdi.scoring import build_feature_matrix, cross_sensor_scores, derivative_sensor_scores
from aeris.models.anomaly import SequenceAutoencoder
from aeris.training.datasets import TemporalDataset
from aeris.training.trainer import seed_everything, select_device


OUTPUT = PROJECT_ROOT / "reports/experiments/step10_2_fdi_multidataset"
KNOWN = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
UNKNOWN = ["SPIKE", "SATURATION", "GAIN_ERROR", "MISSING_BLOCK", "BURST_NOISE"]


def train_autoencoder(X: np.ndarray, device: torch.device, checkpoint: Path) -> tuple[SequenceAutoencoder, float]:
    model = SequenceAutoencoder(input_size=X.shape[2], hidden_size=64).to(device)
    loader = DataLoader(TemporalDataset(X, np.zeros(len(X), dtype=np.float32)), batch_size=128, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    started = time.perf_counter()
    best = float("inf")
    for _ in range(12):
        total = 0.0
        count = 0
        model.train()
        for features, _ in loader:
            features = features.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(features), features)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(features)
            count += len(features)
        if total / count < best:
            best = total / count
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model, time.perf_counter() - started


def ae_sensor_scores(model: nn.Module, X: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(TemporalDataset(np.nan_to_num(X), np.zeros(len(X), dtype=np.float32)), batch_size=128)
    reconstructions = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            reconstructions.append(model(features.to(device)).cpu().numpy())
    global_score, sensor_time_score = reconstruction_scores(np.nan_to_num(X), np.concatenate(reconstructions))
    return global_score, sensor_time_score


def inject_scenarios(
    sources: np.ndarray,
    fault_types: list[str],
    stds: np.ndarray,
    start_index: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    scenario = start_index
    for fault_type in fault_types:
        for source_index in range(min(8, len(sources))):
            sensor = (source_index + len(fault_type)) % sources.shape[2]
            onset = (source_index % 3 + 1) / 4
            severity = (source_index % 3 + 1) / 3
            modified, metadata = inject_fault(
                sources[source_index],
                sensor,
                fault_type,
                onset,
                severity,
                max(float(stds[sensor]), 1e-6),
                seed=42 + scenario,
            )
            windows.append(np.nan_to_num(modified, nan=0.0, posinf=0.0, neginf=0.0))
            metadata.update(
                {
                    "scenario_index": scenario,
                    "source_index": source_index,
                    "known_fault": fault_type in KNOWN,
                    "split": "development" if start_index == 0 else "final_test",
                }
            )
            records.append(metadata)
            scenario += 1
    return np.asarray(windows, dtype=np.float32), pd.DataFrame(records)


def calibrate_threshold(values: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def detector_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, detector: str) -> dict[str, Any]:
    flags = scores >= threshold
    return {
        "detector": detector,
        "threshold": threshold,
        "precision": float(precision_score(labels, flags, zero_division=0)),
        "recall": float(recall_score(labels, flags, zero_division=0)),
        "f1": float(f1_score(labels, flags, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, flags)),
        "mcc": float(matthews_corrcoef(labels, flags)),
        "fpr": float(np.mean(flags[labels == 0])),
        "fnr": float(np.mean(~flags[labels == 1])),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "detection_rate": float(np.mean(flags[labels == 1])),
    }


def rank_rows(scores: np.ndarray, scenarios: pd.DataFrame, detector: str) -> pd.DataFrame:
    rows = []
    for row_index, (_, scenario) in enumerate(scenarios.iterrows()):
        ordering = np.argsort(scores[row_index])[::-1]
        rank = int(np.where(ordering == int(scenario["sensor_index"]))[0][0] + 1)
        rows.append(
            {
                "detector": detector,
                "scenario_index": int(scenario["scenario_index"]),
                "fault_type": scenario["fault_type"],
                "sensor_index": int(scenario["sensor_index"]),
                "rank": rank,
                "top1": rank <= 1,
                "top3": rank <= 3,
                "top5": rank <= 5,
                "mrr": 1.0 / rank,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    seed_everything(42)
    data = load_prepared_cmapss(PROJECT_ROOT / "data/processed/cmapss_fd001")
    X_train = np.nan_to_num(data["X_train"]).astype(np.float32)
    X_validation = np.nan_to_num(data["X_validation"]).astype(np.float32)
    calibration, final_normal = np.array_split(X_validation, 2)
    device = select_device()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model, training_seconds = train_autoencoder(
        X_train, device, OUTPUT / "models" / "sequence_autoencoder.pt"
    )
    baseline = fit_robust_baseline(X_train)
    stds = np.std(X_train, axis=(0, 1))

    development_faults, development_manifest = inject_scenarios(calibration, KNOWN, stds, 0)
    final_faults, final_manifest = inject_scenarios(final_normal, KNOWN + UNKNOWN, stds, len(development_manifest))

    def feature_scores(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        robust_sensor = robust_sensor_scores(windows, baseline)
        derivative_sensor = derivative_sensor_scores(windows)
        cross_sensor = cross_sensor_scores(windows, np.std(X_train, axis=(0, 1)))
        return robust_sensor, derivative_sensor, cross_sensor

    calibration_features = feature_scores(calibration)
    final_normal_features = feature_scores(final_normal)
    development_features = feature_scores(development_faults)
    final_fault_features = feature_scores(final_faults)
    calibration_ae, calibration_ae_sensor = ae_sensor_scores(model, calibration, device)
    normal_ae, normal_ae_sensor = ae_sensor_scores(model, final_normal, device)
    development_ae, development_ae_sensor = ae_sensor_scores(model, development_faults, device)
    final_ae, final_ae_sensor = ae_sensor_scores(model, final_faults, device)

    thresholds = {}
    for percentile in (90.0, 95.0, 97.5, 99.0):
        thresholds[str(percentile)] = {
            "statistical": calibrate_threshold(calibration_features[0].max(axis=1), percentile),
            "autoencoder": calibrate_threshold(calibration_ae, percentile),
            "temporal": calibrate_threshold(calibration_features[1].max(axis=1), percentile),
            "cross_sensor": calibrate_threshold(calibration_features[2].max(axis=1), percentile),
        }
    selected = thresholds["95.0"]
    normal_scores = {
        "statistical": final_normal_features[0].max(axis=1),
        "autoencoder": normal_ae,
        "temporal": final_normal_features[1].max(axis=1),
        "cross_sensor": final_normal_features[2].max(axis=1),
    }
    fault_scores = {
        "statistical": final_fault_features[0].max(axis=1),
        "autoencoder": final_ae,
        "temporal": final_fault_features[1].max(axis=1),
        "cross_sensor": final_fault_features[2].max(axis=1),
    }
    detector_sensor = {
        "statistical": final_fault_features[0],
        "autoencoder": final_ae_sensor,
        "temporal": final_fault_features[1],
        "cross_sensor": final_fault_features[2],
    }
    normalized = {key: fault_scores[key] / max(selected[key], 1e-9) for key in selected}
    normal_normalized = {key: normal_scores[key] / max(selected[key], 1e-9) for key in selected}
    scores = {
        "statistical": np.concatenate([normal_scores["statistical"], fault_scores["statistical"]]),
        "autoencoder": np.concatenate([normal_scores["autoencoder"], fault_scores["autoencoder"]]),
        "temporal": np.concatenate([normal_scores["temporal"], fault_scores["temporal"]]),
        "cross_sensor": np.concatenate([normal_scores["cross_sensor"], fault_scores["cross_sensor"]]),
        "hybrid": np.concatenate(
            [sum(normal_normalized.values()), sum(normalized.values())]
        ),
    }
    labels = np.concatenate([np.zeros(len(final_normal), dtype=int), np.ones(len(final_faults), dtype=int)])
    threshold_by_detector = {key: selected.get(key, 3.0) for key in scores if key != "hybrid"}
    threshold_by_detector["hybrid"] = 3.0
    results = pd.DataFrame(
        [detector_metrics(labels, values, threshold_by_detector[name], name) for name, values in scores.items()]
    )

    isolation = pd.concat(
        [rank_rows(values, final_manifest, name) for name, values in detector_sensor.items()],
        ignore_index=True,
    )
    isolation.to_csv(OUTPUT / "isolation_results.csv", index=False)
    detection = pd.DataFrame(
        [
            {
                "scenario_index": int(row["scenario_index"]),
                "fault_type": row["fault_type"],
                "known_fault": bool(row["known_fault"]),
                "anomaly_score": float(scores["hybrid"][len(final_normal) + index]),
                "detected": bool(scores["hybrid"][len(final_normal) + index] >= 3.0),
            }
            for index, (_, row) in enumerate(final_manifest.iterrows())
        ]
    )
    detection.to_csv(OUTPUT / "detection_results.csv", index=False)

    classifier_train_X = build_feature_matrix(
        development_faults,
        development_features[0],
        development_features[1],
        development_features[2],
    )
    classifier_test_X = build_feature_matrix(
        final_faults,
        final_fault_features[0],
        final_fault_features[1],
        final_fault_features[2],
    )
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    classifier.fit(classifier_train_X, development_manifest["fault_type"])
    predicted_types = classifier.predict(classifier_test_X)
    classification = pd.DataFrame(
        {
            "scenario_index": final_manifest["scenario_index"],
            "actual_fault_type": final_manifest["fault_type"],
            "predicted_fault_type": predicted_types,
            "known_fault": final_manifest["known_fault"],
            "unknown_flag": ~final_manifest["known_fault"],
        }
    )
    classification.to_csv(OUTPUT / "fault_classification_results.csv", index=False)
    known_mask = classification["known_fault"].to_numpy()
    unknown_predictions = ~np.isin(predicted_types, KNOWN)
    unknown_metrics = {
        "unknown_fault_count": int(np.sum(~known_mask)),
        "unknown_detection_recall": float(np.mean(unknown_predictions[~known_mask])),
        "unknown_precision": float(
            np.sum(unknown_predictions & ~known_mask) / max(1, np.sum(unknown_predictions))
        ),
        "known_unknown_confusion_matrix": confusion_matrix(
            ~known_mask, unknown_predictions, labels=[False, True]
        ).tolist(),
    }
    classification_report = {
        "known_fault_macro_f1": float(
            f1_score(
                classification.loc[known_mask, "actual_fault_type"],
                classification.loc[known_mask, "predicted_fault_type"],
                average="macro",
                zero_division=0,
            )
        ),
        **unknown_metrics,
    }
    (OUTPUT / "fault_classification_results.json").write_text(
        json.dumps(classification_report, indent=2), encoding="utf-8"
    )

    ablation = []
    for name in ("statistical", "autoencoder", "temporal", "cross_sensor", "hybrid"):
        row = results.loc[results["detector"] == name].iloc[0].to_dict()
        row["ablation_component"] = name
        ablation.append(row)
    pd.DataFrame(ablation).to_csv(OUTPUT / "ablation_results.csv", index=False)
    thresholds_frame = pd.DataFrame(
        [{"percentile": percentile, **values} for percentile, values in thresholds.items()]
    )
    thresholds_frame.to_csv(OUTPUT / "threshold_analysis.csv", index=False)

    leakage_audit = {
        "training_dataset": "C-MAPSS FD001 X_train",
        "calibration_dataset": "C-MAPSS FD001 first half of X_validation",
        "final_normal_dataset": "C-MAPSS FD001 second half of X_validation",
        "development_fault_scenarios": int(len(development_manifest)),
        "final_fault_scenarios": int(len(final_manifest)),
        "scenario_overlap": False,
        "fault_training_types": KNOWN,
        "unknown_test_types": UNKNOWN,
        "thresholds_fit_on_final_labels": False,
        "prepared_arrays_refit": False,
    }
    (OUTPUT / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2), encoding="utf-8")
    development_manifest.to_csv(OUTPUT / "scenario_manifest_development.csv", index=False)
    final_manifest.to_csv(OUTPUT / "scenario_manifest_final.csv", index=False)
    (OUTPUT / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "available": ["C-MAPSS FD001"],
                "unavailable": ["FD002", "FD003", "FD004", "ESA anomaly dataset"],
                "reason": "Not present locally; no unverified downloads or fabricated results.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "split_manifest.json").write_text(json.dumps(leakage_audit, indent=2), encoding="utf-8")
    (OUTPUT / "model_configs.json").write_text(
        json.dumps(
            {
                "autoencoder": {"type": "LSTM sequence autoencoder", "hidden_size": 64, "epochs": 12},
                "classifier": {"type": "RandomForestClassifier", "estimators": 100, "seed": 42},
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results.to_csv(OUTPUT / "results.csv", index=False)
    results.to_json(OUTPUT / "results.json", orient="records", indent=2)
    plots = OUTPUT / "plots"
    plots.mkdir(exist_ok=True)
    for name, values in scores.items():
        plt.figure(figsize=(6, 4))
        plt.hist(values[labels == 0], bins=30, alpha=0.6, label="normal degradation")
        plt.hist(values[labels == 1], bins=30, alpha=0.6, label="fault scenarios")
        plt.axvline(threshold_by_detector[name], color="black", linestyle="--", label="threshold")
        plt.title(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / f"{name}_score_distribution.png", dpi=150)
        plt.close()

    report = f"""# Step 10.2 Multi-Dataset FDI Upgrade

## Scope and data availability

This is an isolated upgrade of the historical Step 10.2 experiment. The
authoritative roadmap was not changed: Step 7 remains the current authorized
milestone. Only C-MAPSS FD001 was available locally. FD002-FD004 and the ESA
anomaly dataset were not used, so cross-condition and cross-dataset results
are unavailable rather than fabricated.

## Architecture

The pipeline is `quality-aware telemetry -> robust statistical, temporal,
cross-sensor, and autoencoder scores -> hybrid decision -> sensor isolation ->
fault classifier -> open-set analysis`. The autoencoder and classifier were
trained only on development/normal data. Final scenarios use distinct source
windows and scenario identities.

## Final locked evaluation

{results.to_string(index=False)}

## Fault classification and open set

{json.dumps(classification_report, indent=2)}

Unknown types were absent from classifier training: {", ".join(UNKNOWN)}.
The classifier is not allowed to call an unseen label unknown without an
explicit rejection policy; therefore the reported unknown metrics measure
actual rejection behavior and may be zero.

## Isolation

The full per-scenario isolation table is in `isolation_results.csv`, including
Top-1, Top-3, Top-5, MRR, rank, and sensor identity. No single aggregate is
presented as proof of successful isolation.

## Leakage audit

See `leakage_audit.json`. Training uses FD001 X_train, calibration uses only
the first half of normal validation, and final normal/fault evaluation uses
the second half with disjoint scenario identities. Thresholds were selected
without final fault labels. Existing scalers and prepared arrays were not
refit.

## Cost and limitations

Device: `{device}`. Autoencoder training time: `{training_seconds:.2f}` seconds.
The detector remains a controlled synthetic-fault study. Natural degradation
is represented by held-out normal FD001 validation windows, but this is not
equivalent to operational fleet telemetry. Cross-condition, cross-dataset,
and real-fault conclusions remain unavailable.
"""
    (OUTPUT / "step10_2_fdi_multidataset_report.md").write_text(report, encoding="utf-8")
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    (OUTPUT / "config.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "known_faults": KNOWN,
                "unknown_faults": UNKNOWN,
                "threshold_percentiles": [90.0, 95.0, 97.5, 99.0],
                "selected_percentile": 95.0,
                "device": str(device),
                "training_seconds": training_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
