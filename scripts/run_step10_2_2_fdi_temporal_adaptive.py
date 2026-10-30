"""Execute the isolated Step 10.2.2 temporal adaptive FDI experiment."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.adapters.cmapss_adapter import load_prepared_cmapss
from aeris.evaluation.anomaly import fit_robust_baseline
from aeris.faults.injection import inject_fault
from aeris.fdi.adaptive import ewma, soft_evidence, temporal_sensor_features
from aeris.training.trainer import seed_everything, select_device


OUT = ROOT / "reports/experiments/step10_2_2_fdi_temporal_adaptive"
KNOWN = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
UNKNOWN = ["SPIKE", "SATURATION", "GAIN_ERROR", "DEADBAND", "BURST_NOISE", "COMPOSITE"]


def scenarios(sources: np.ndarray, types: list[str], std: np.ndarray, start: int):
    windows = []
    rows = []
    scenario_id = start
    for fault_type in types:
        for source_index in range(min(8, len(sources))):
            sensor = (source_index * 5 + len(fault_type)) % sources.shape[2]
            severity_name = ("LOW", "MEDIUM", "HIGH")[source_index % 3]
            severity = (0.3, 0.6, 1.0)[source_index % 3]
            onset = (0.2, 0.45, 0.7)[source_index % 3]
            modified, metadata = inject_fault(
                sources[source_index],
                sensor,
                fault_type,
                onset,
                severity,
                max(float(std[sensor]), 1e-6),
                seed=100 + scenario_id,
            )
            if source_index % 4 == 0:
                second = (sensor + 1) % sources.shape[2]
                modified, _ = inject_fault(
                    modified, second, "BIAS", onset, severity,
                    max(float(std[second]), 1e-6), seed=1000 + scenario_id,
                )
                target_sensors = [sensor, second]
            else:
                target_sensors = [sensor]
            windows.append(np.nan_to_num(modified, nan=0.0, posinf=0.0, neginf=0.0))
            metadata.update(
                {
                    "scenario_id": scenario_id,
                    "source_index": source_index,
                    "engine_id": "validation_window",
                    "sensor_id": sensor,
                    "target_sensors": json.dumps(target_sensors),
                    "severity": severity_name,
                    "onset_cycle": int(onset * sources.shape[1]),
                    "duration": int(sources.shape[1] * (1 - onset)),
                    "persistence": "SHORT_TRANSIENT" if source_index == 0 else "PERSISTENT",
                    "random_seed": 100 + scenario_id,
                    "known_fault": fault_type in KNOWN,
                }
            )
            rows.append(metadata)
            scenario_id += 1
    return np.asarray(windows, dtype=np.float32), pd.DataFrame(rows)


def score(windows: np.ndarray, baseline: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    features = temporal_sensor_features(windows, baseline["median"], baseline["scale"])
    sensor = soft_evidence(features)
    return np.max(sensor, axis=1), sensor


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    flags = scores >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, flags, zero_division=0)),
        "recall": float(recall_score(labels, flags, zero_division=0)),
        "f1": float(f1_score(labels, flags, zero_division=0)),
        "fpr": float(np.mean(flags[labels == 0])),
        "fnr": float(np.mean(~flags[labels == 1])),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "detection_rate": float(np.mean(flags[labels == 1])),
    }


def main() -> None:
    started = time.perf_counter()
    seed_everything(42)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_prepared_cmapss(ROOT / "data/processed/cmapss_fd001")
    X_train = np.nan_to_num(data["X_train"]).astype(np.float32)
    X_validation = np.nan_to_num(data["X_validation"]).astype(np.float32)
    calibration, final_normal = np.array_split(X_validation, 2)
    baseline = fit_robust_baseline(X_train)
    std = np.std(X_train, axis=(0, 1))
    development, development_manifest = scenarios(calibration, KNOWN, std, 0)
    final_faults, final_manifest = scenarios(final_normal, KNOWN + UNKNOWN, std, len(development_manifest))

    calibration_score, _ = score(calibration, baseline)
    development_score, development_sensor = score(development, baseline)
    normal_score, normal_sensor = score(final_normal, baseline)
    fault_score, fault_sensor = score(final_faults, baseline)
    dev_labels = np.ones(len(development_score), dtype=int)
    thresholds = np.unique(np.percentile(np.concatenate([calibration_score, development_score]), np.linspace(80, 99, 50)))
    threshold_rows = []
    for threshold in thresholds:
        row = metrics(np.concatenate([np.zeros(len(calibration_score)), dev_labels]), np.concatenate([calibration_score, development_score]), threshold)
        row["threshold"] = float(threshold)
        threshold_rows.append(row)
    threshold_frame = pd.DataFrame(threshold_rows)
    eligible = threshold_frame[threshold_frame["fpr"] <= 0.10]
    threshold = float(eligible.sort_values(["f1", "recall"], ascending=False).iloc[0]["threshold"]) if not eligible.empty else float(thresholds[-1])

    labels = np.concatenate([np.zeros(len(normal_score), dtype=int), np.ones(len(fault_score), dtype=int)])
    final_scores = np.concatenate([normal_score, fault_score])
    detection = metrics(labels, final_scores, threshold)
    detection["detector"] = "soft_temporal_adaptive"
    detection["training_time_seconds"] = time.perf_counter() - started
    calibration_features = temporal_sensor_features(calibration, baseline["median"], baseline["scale"])
    development_features = temporal_sensor_features(development, baseline["median"], baseline["scale"])
    normal_features = temporal_sensor_features(final_normal, baseline["median"], baseline["scale"])
    fault_features = temporal_sensor_features(final_faults, baseline["median"], baseline["scale"])
    robust_calibration = np.max(calibration_features[:, :, 3] / 12.0, axis=1)
    robust_development = np.max(development_features[:, :, 3] / 12.0, axis=1)
    robust_normal = np.max(normal_features[:, :, 3] / 12.0, axis=1)
    robust_fault = np.max(fault_features[:, :, 3] / 12.0, axis=1)
    robust_threshold = float(np.percentile(np.concatenate([robust_calibration, robust_development]), 95))
    robust_detection = metrics(
        labels,
        np.concatenate([robust_normal, robust_fault]),
        robust_threshold,
    )
    robust_detection.update({"detector": "adaptive_robust_z", "training_time_seconds": detection["training_time_seconds"]})

    # Dedicated sensor ranking models trained only on development scenarios.
    train_features = temporal_sensor_features(development, baseline["median"], baseline["scale"])
    test_features = temporal_sensor_features(final_faults, baseline["median"], baseline["scale"])
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    train_targets = np.concatenate([np.full(train_features.shape[1], row["sensor_id"]) for _, row in development_manifest.iterrows()])
    models = {
        "random_forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42),
    }
    isolation_rows = []
    isolation_metrics = []
    for name, model in models.items():
        model.fit(train_flat, train_targets)
        joblib.dump(model, OUT / f"{name}_sensor_classifier.joblib")
        for index, (_, row) in enumerate(final_manifest.iterrows()):
            probabilities = model.predict_proba(test_features[index])
            ranking = np.argsort(probabilities.max(axis=1))[::-1]
            targets = set(json.loads(row["target_sensors"]))
            rank = min(int(np.where(ranking == target)[0][0] + 1) for target in targets)
            isolation_rows.append({"model": name, "scenario_id": row["scenario_id"], "fault_type": row["fault_type"], "rank": rank, "top1": rank <= 1, "top3": rank <= 3, "top5": rank <= 5, "mrr": 1 / rank})
        subset = pd.DataFrame([item for item in isolation_rows if item["model"] == name])
        isolation_metrics.append({"model": name, "top1": subset["top1"].mean(), "top3": subset["top3"].mean(), "top5": subset["top5"].mean(), "mrr": subset["mrr"].mean(), "mean_rank": subset["rank"].mean()})
    isolation_frame = pd.DataFrame(isolation_rows)

    # Explicit open-set distance from known fault centroids.
    classifier = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    train_fault_features = train_features.mean(axis=1)
    test_fault_features = test_features.mean(axis=1)
    classifier.fit(train_fault_features, development_manifest["fault_type"])
    probabilities = classifier.predict_proba(test_fault_features)
    predicted = classifier.classes_[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    known_confidence = classifier.predict_proba(train_fault_features).max(axis=1)
    rejection_threshold = float(np.percentile(known_confidence, 5))
    rejected = confidence < rejection_threshold
    known_mask = final_manifest["known_fault"].to_numpy()
    open_set = {
        "known_macro_f1": float(f1_score(final_manifest.loc[known_mask, "fault_type"], predicted[known_mask], average="macro", zero_division=0)),
        "unknown_recall": float(np.mean(rejected[~known_mask])),
        "unknown_precision": float(np.sum(rejected & ~known_mask) / max(1, np.sum(rejected))),
        "unknown_f1": float(f1_score(~known_mask, rejected, zero_division=0)),
        "unknown_auroc": float(roc_auc_score(~known_mask, 1 - confidence)),
        "unknown_aupr": float(average_precision_score(~known_mask, 1 - confidence)),
        "false_acceptance_rate": float(np.mean(~rejected[~known_mask])),
    }
    classification = pd.DataFrame({"scenario_id": final_manifest["scenario_id"], "actual_fault_type": final_manifest["fault_type"], "predicted_fault_type": predicted, "confidence": confidence, "rejected_unknown": rejected})

    results = pd.DataFrame([robust_detection, detection])
    results.to_csv(OUT / "results.csv", index=False)
    results.to_json(OUT / "results.json", orient="records", indent=2)
    isolation_frame.to_csv(OUT / "isolation_results.csv", index=False)
    classification.to_csv(OUT / "fault_classification_results.csv", index=False)
    pd.DataFrame([open_set]).to_csv(OUT / "open_set_results.csv", index=False)
    threshold_frame.to_csv(OUT / "threshold_analysis.csv", index=False)
    pd.DataFrame(isolation_metrics).to_csv(OUT / "ablation_results.csv", index=False)
    development_manifest.to_csv(OUT / "scenario_manifest_development.csv", index=False)
    final_manifest.to_csv(OUT / "scenario_manifest.csv", index=False)
    pd.DataFrame({"score": final_scores, "label": labels, "detected": final_scores >= threshold}).to_csv(OUT / "detection_results.csv", index=False)
    pd.DataFrame({"engine_id": ["validation_window"] * len(final_normal), "false_alarm": final_scores[:len(final_normal)] >= threshold}).to_csv(OUT / "false_positive_analysis.csv", index=False)
    (OUT / "dataset_manifest.json").write_text(json.dumps({"used": ["C-MAPSS FD001"], "unavailable": ["FD002", "FD003", "FD004", "ESA anomaly dataset"]}, indent=2), encoding="utf-8")
    (OUT / "leakage_audit.json").write_text(json.dumps({"training": "normal X_train plus development known faults", "calibration": "normal calibration plus development scenarios", "final": "disjoint normal and known/unknown scenarios", "scenario_overlap": False, "unknown_faults_in_training": [], "final_labels_used_for_threshold": False}, indent=2), encoding="utf-8")
    (OUT / "calibration_results.csv").write_text(pd.DataFrame([{"nominal": 0.95, "observed": 1 - detection["fpr"], "note": "Detection confidence calibration is not a coverage guarantee."}]).to_csv(index=False), encoding="utf-8")
    (OUT / "fault_confusion_matrix.csv").write_text(pd.DataFrame(confusion_matrix(final_manifest["fault_type"], predicted, labels=sorted(set(KNOWN + UNKNOWN))), index=sorted(set(KNOWN + UNKNOWN)), columns=sorted(set(KNOWN + UNKNOWN))).to_csv(), encoding="utf-8")
    for filename, curve in (("roc_curves.png", roc_curve), ("precision_recall_curves.png", precision_recall_curve)):
        plt.figure(figsize=(6, 4))
        x, y, _ = curve(labels, final_scores)
        plt.plot(x, y)
        plt.tight_layout()
        plt.savefig(OUT / filename, dpi=150)
        plt.close()
    report = f"""# Step 10.2.2 Temporal Sensor Isolation and Adaptive FDI

## Executive summary

This isolated experiment tests causal temporal features, adaptive soft
evidence, dedicated sensor classifiers, and explicit open-set rejection.
Only C-MAPSS FD001 was available. No roadmap state or historical experiment
was changed.

## Final detection

{json.dumps(detection, indent=2)}

## Sensor isolation

{json.dumps(isolation_metrics, indent=2)}

## Fault classification and open set

{json.dumps(open_set, indent=2)}

## Leakage controls

Training uses normal training telemetry and development known-fault scenarios.
Calibration uses normal calibration and development scenarios. Final scenarios
use disjoint windows, seeds, and unknown fault types. Final labels were not
used for threshold selection.

## Interpretation

The detector uses soft evidence rather than the hard persistence gate that
collapsed Step 10.2.1 recall. These metrics are a measured result, not a
success claim. Cross-dataset and real-fault validation remain unavailable.
"""
    (OUT / "step10_2_2_fdi_report.md").write_text(report, encoding="utf-8")
    (OUT / "optimization_report.md").write_text(report, encoding="utf-8")
    print(results.to_string(index=False))
    print(json.dumps(open_set, indent=2))


if __name__ == "__main__":
    main()
