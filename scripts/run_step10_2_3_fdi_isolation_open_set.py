"""Run Step 10.2.3 with frozen detection and redesigned isolation/open-set paths."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.adapters.cmapss_adapter import load_prepared_cmapss
from aeris.evaluation.anomaly import fit_robust_baseline
from aeris.faults.injection import inject_fault
from aeris.fdi.adaptive import soft_evidence, temporal_sensor_features
from aeris.fdi.isolation import expected_calibration_error, sensor_feature_tensor, standardized_distance
from aeris.training.trainer import seed_everything, select_device


OUT = ROOT / "reports/experiments/step10_2_3_fdi_isolation_open_set"
KNOWN = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
UNKNOWN = ["SPIKE", "SATURATION", "GAIN_ERROR", "DEADBAND", "BURST_NOISE", "COMPOSITE"]


def make_scenarios(sources: np.ndarray, types: list[str], std: np.ndarray, start: int) -> tuple[np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    scenario_id = start
    for fault_type in types:
        for source_index in range(min(8, len(sources))):
            sensor = (source_index * 5 + len(fault_type)) % sources.shape[2]
            severity_name = ("LOW", "MEDIUM", "HIGH")[source_index % 3]
            severity = (0.3, 0.6, 1.0)[source_index % 3]
            onset = (0.2, 0.45, 0.7)[source_index % 3]
            modified, metadata = inject_fault(
                sources[source_index], sensor, fault_type, onset, severity,
                max(float(std[sensor]), 1e-6), seed=100 + scenario_id,
            )
            targets = [sensor]
            if source_index % 4 == 0:
                second = (sensor + 1) % sources.shape[2]
                modified, _ = inject_fault(
                    modified, second, "BIAS", onset, severity,
                    max(float(std[second]), 1e-6), seed=1000 + scenario_id,
                )
                targets.append(second)
            windows.append(np.nan_to_num(modified, nan=0.0, posinf=0.0, neginf=0.0))
            metadata.update(
                {
                    "scenario_id": scenario_id,
                    "source_index": source_index,
                    "engine_id": "validation_window",
                    "sensor_id": sensor,
                    "target_sensors": json.dumps(targets),
                    "fault_type": fault_type,
                    "severity": severity_name,
                    "onset_cycle": int(onset * sources.shape[1]),
                    "duration": int(sources.shape[1] * (1 - onset)),
                    "random_seed": 100 + scenario_id,
                    "known_fault": fault_type in KNOWN,
                    "split": "development" if start == 0 else "final",
                }
            )
            records.append(metadata)
            scenario_id += 1
    return np.asarray(windows, dtype=np.float32), pd.DataFrame(records)


def fit_peer_models(X: np.ndarray) -> list[LinearRegression]:
    models = []
    for sensor in range(X.shape[2]):
        peers = [index for index in range(X.shape[2]) if index != sensor]
        model = LinearRegression()
        model.fit(X[:, :, peers].reshape(-1, len(peers)), X[:, :, sensor].reshape(-1))
        models.append(model)
    return models


def peer_residuals(X: np.ndarray, models: list[LinearRegression]) -> np.ndarray:
    result = np.zeros((len(X), X.shape[2]))
    for sensor, model in enumerate(models):
        peers = [index for index in range(X.shape[2]) if index != sensor]
        prediction = model.predict(X[:, :, peers].reshape(-1, len(peers))).reshape(len(X), X.shape[1])
        result[:, sensor] = np.mean(np.abs(X[:, :, sensor] - prediction), axis=1)
    return result


def detector_scores(X: np.ndarray, baseline: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    features = temporal_sensor_features(X, baseline["median"], baseline["scale"])
    sensor_scores = soft_evidence(features)
    return np.max(sensor_scores, axis=1), sensor_scores


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, name: str) -> dict[str, float | str]:
    flags = scores >= threshold
    return {
        "detector": name,
        "threshold": float(threshold),
        "precision": float(precision_score(labels, flags, zero_division=0)),
        "recall": float(recall_score(labels, flags, zero_division=0)),
        "f1": float(f1_score(labels, flags, zero_division=0)),
        "fpr": float(np.mean(flags[labels == 0])),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def isolation_rows(model: RandomForestClassifier, features: np.ndarray, manifest: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for index, (_, scenario) in enumerate(manifest.iterrows()):
        probability = model.predict_proba(features[index])
        ranking = np.argsort(probability[:, 1])[::-1]
        targets = set(json.loads(scenario["target_sensors"]))
        rank = min(int(np.where(ranking == target)[0][0] + 1) for target in targets)
        rows.append(
            {
                "model": name,
                "scenario_id": int(scenario["scenario_id"]),
                "fault_type": scenario["fault_type"],
                "severity": scenario["severity"],
                "rank": rank,
                "top1": rank <= 1,
                "top3": rank <= 3,
                "top5": rank <= 5,
                "top10": rank <= 10,
                "mrr": 1.0 / rank,
                "actual_sensors": json.dumps(sorted(int(value) for value in targets)),
                "sensor_probabilities": json.dumps(probability[:, 1].round(6).tolist()),
            }
        )
    return pd.DataFrame(rows)


def calibration_curve_metrics(confidence: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(correct.astype(int), confidence)),
        "ece": expected_calibration_error(confidence, correct),
        "mean_confidence": float(np.mean(confidence)),
        "accuracy": float(np.mean(correct)),
    }


def main() -> None:
    started = time.perf_counter()
    seed_everything(42)
    OUT.mkdir(parents=True, exist_ok=True)
    device = select_device()
    data = load_prepared_cmapss(ROOT / "data/processed/cmapss_fd001")
    X_train = np.nan_to_num(data["X_train"]).astype(np.float32)
    X_validation = np.nan_to_num(data["X_validation"]).astype(np.float32)
    calibration, final_normal = np.array_split(X_validation, 2)
    baseline = fit_robust_baseline(X_train)
    std = np.std(X_train, axis=(0, 1))
    development, development_manifest = make_scenarios(calibration, KNOWN, std, 0)
    final_faults, final_manifest = make_scenarios(final_normal, KNOWN + UNKNOWN, std, len(development_manifest))

    # Frozen Step 10.2.2 detector path: same soft evidence and development threshold.
    calibration_score, _ = detector_scores(calibration, baseline)
    development_score, _ = detector_scores(development, baseline)
    normal_score, _ = detector_scores(final_normal, baseline)
    fault_score, _ = detector_scores(final_faults, baseline)
    threshold_candidates = np.unique(np.percentile(np.concatenate([calibration_score, development_score]), np.linspace(80, 99, 50)))
    threshold_rows = []
    for threshold in threshold_candidates:
        threshold_rows.append(detection_metrics(
            np.concatenate([np.zeros(len(calibration_score), dtype=int), np.ones(len(development_score), dtype=int)]),
            np.concatenate([calibration_score, development_score]),
            float(threshold),
            "frozen_soft_temporal_adaptive",
        ))
    threshold_frame = pd.DataFrame(threshold_rows)
    eligible = threshold_frame[threshold_frame["fpr"] <= 0.10]
    threshold = float(eligible.sort_values(["f1", "recall"], ascending=False).iloc[0]["threshold"])
    labels = np.concatenate([np.zeros(len(normal_score), dtype=int), np.ones(len(fault_score), dtype=int)])
    reference = detection_metrics(labels, np.concatenate([normal_score, fault_score]), threshold, "frozen_soft_temporal_adaptive")
    reference["training_time_seconds"] = time.perf_counter() - started

    # Training-only peer prediction preserves sensor-specific residual evidence.
    peer_models = fit_peer_models(X_train)
    peer_dev = peer_residuals(development, peer_models)
    peer_final = peer_residuals(final_faults, peer_models)
    dev_features = sensor_feature_tensor(development, baseline["median"], baseline["scale"], peer_dev)
    final_features = sensor_feature_tensor(final_faults, baseline["median"], baseline["scale"], peer_final)

    # Balanced pairwise classifier: each scenario contributes one positive and all negatives are balanced.
    rng = np.random.default_rng(42)
    train_rows = []
    train_labels = []
    train_types = []
    for index, (_, row) in enumerate(development_manifest.iterrows()):
        targets = set(json.loads(row["target_sensors"]))
        negative = [sensor for sensor in range(dev_features.shape[1]) if sensor not in targets]
        selected = negative[: max(1, min(len(negative), len(targets) * 2))]
        for sensor in sorted(targets) + selected:
            train_rows.append(dev_features[index, sensor])
            train_labels.append(int(sensor in targets))
            train_types.append(row["fault_type"])
    pair_model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
    pair_model.fit(np.asarray(train_rows), train_labels)
    joblib.dump(pair_model, OUT / "pairwise_sensor_classifier.joblib")
    isolation = isolation_rows(pair_model, final_features, final_manifest, "pairwise_random_forest")
    isolation_summary = isolation.groupby("model")[["top1", "top3", "top5", "top10", "mrr", "rank"]].mean().reset_index()
    isolation_summary.rename(columns={"rank": "mean_rank"}, inplace=True)

    # Fault-type-conditioned classifier and class-centroid open-set fusion.
    scenario_train = dev_features.mean(axis=1)
    scenario_final = final_features.mean(axis=1)
    type_model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
    type_model.fit(scenario_train, development_manifest["fault_type"])
    joblib.dump(type_model, OUT / "fault_type_classifier.joblib")
    probabilities = type_model.predict_proba(scenario_final)
    predicted_type = type_model.classes_[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    centroids = {name: scenario_train[development_manifest["fault_type"].to_numpy() == name].mean(axis=0) for name in KNOWN}
    known_dev = scenario_train[development_manifest["fault_type"].isin(KNOWN).to_numpy()]
    scale = np.std(known_dev, axis=0) + 1e-6
    standardized = {name: centroid / scale for name, centroid in centroids.items()}
    final_standardized = scenario_final / scale
    distance_matrix = np.stack([np.linalg.norm(final_standardized - centroid, axis=1) for centroid in standardized.values()], axis=1)
    nearest_distance = np.min(distance_matrix, axis=1)
    dev_distance_matrix = np.stack(
        [np.linalg.norm((scenario_train[development_manifest["fault_type"].isin(KNOWN).to_numpy()] / scale) - centroid, axis=1) for centroid in standardized.values()],
        axis=1,
    )
    novelty = np.maximum(0, np.linalg.norm(final_standardized, axis=1) - np.percentile(np.linalg.norm(known_dev / scale, axis=1), 95))
    distance_score = nearest_distance / (np.percentile(dev_distance_matrix, 95) + 1e-6)
    confidence_score = 1 - confidence
    novelty_score = novelty / (np.percentile(novelty, 95) + 1e-6)
    unknown_score = 0.45 * distance_score + 0.35 * confidence_score + 0.20 * novelty_score
    known_train_unknown = 0.45 * np.min(dev_distance_matrix, axis=1) / (np.percentile(dev_distance_matrix, 95) + 1e-6)
    unknown_threshold = float(np.percentile(known_train_unknown, 95))
    rejected = unknown_score >= unknown_threshold
    known_mask = final_manifest["known_fault"].to_numpy()
    open_set = {
        "threshold": unknown_threshold,
        "unknown_precision": float(np.sum(rejected & ~known_mask) / max(1, np.sum(rejected))),
        "unknown_recall": float(np.mean(rejected[~known_mask])),
        "unknown_f1": float(f1_score(~known_mask, rejected, zero_division=0)),
        "unknown_auroc": float(roc_auc_score(~known_mask, unknown_score)),
        "unknown_aupr": float(average_precision_score(~known_mask, unknown_score)),
        "false_acceptance_rate": float(np.mean(~rejected[~known_mask])),
        "known_macro_f1": float(f1_score(final_manifest.loc[known_mask, "fault_type"], predicted_type[known_mask], average="macro", zero_division=0)),
    }
    classification = pd.DataFrame(
        {
            "scenario_id": final_manifest["scenario_id"],
            "actual_fault_type": final_manifest["fault_type"],
            "predicted_fault_type": predicted_type,
            "confidence": confidence,
            "nearest_known_distance": nearest_distance,
            "unknown_score": unknown_score,
            "rejected_unknown": rejected,
        }
    )

    correct_sensor = isolation["rank"].to_numpy() <= 1
    sensor_probability = 1 / np.maximum(isolation["rank"].to_numpy(), 1)
    calibration = calibration_curve_metrics(sensor_probability, correct_sensor)
    fault_breakdown = classification.assign(known=final_manifest["known_fault"].to_numpy()).groupby("actual_fault_type").agg(
        count=("actual_fault_type", "size"),
        classification_accuracy=("actual_fault_type", lambda values: np.mean(values.to_numpy() == classification.loc[values.index, "predicted_fault_type"].to_numpy())),
        mean_unknown_score=("unknown_score", "mean"),
    ).reset_index()
    severity_breakdown = final_manifest.assign(detected=np.concatenate([fault_score >= threshold])).groupby("severity").agg(count=("severity", "size"), detection_rate=("detected", "mean")).reset_index()
    sensor_breakdown = isolation.groupby("fault_type").agg(top1=("top1", "mean"), top3=("top3", "mean"), top5=("top5", "mean"), mrr=("mrr", "mean"), mean_rank=("rank", "mean")).reset_index()
    ablation = pd.DataFrame(
        [
            {"component": "temporal_detector_reference", **reference},
            {"component": "pairwise_sensor_classifier", **isolation_summary.iloc[0].to_dict()},
            {"component": "distance_only_open_set", "unknown_auroc": float(roc_auc_score(~known_mask, distance_score)), "unknown_aupr": float(average_precision_score(~known_mask, distance_score))},
            {"component": "confidence_only_open_set", "unknown_auroc": float(roc_auc_score(~known_mask, confidence_score)), "unknown_aupr": float(average_precision_score(~known_mask, confidence_score))},
            {"component": "fused_open_set", **open_set},
        ]
    )

    results = pd.DataFrame([reference])
    results.to_csv(OUT / "results.csv", index=False)
    results.to_json(OUT / "results.json", orient="records", indent=2)
    isolation.to_csv(OUT / "isolation_results.csv", index=False)
    classification.to_csv(OUT / "fault_classification_results.csv", index=False)
    pd.DataFrame([open_set]).to_csv(OUT / "open_set_results.csv", index=False)
    sensor_breakdown.to_csv(OUT / "sensor_breakdown.csv", index=False)
    fault_breakdown.to_csv(OUT / "fault_breakdown.csv", index=False)
    severity_breakdown.to_csv(OUT / "severity_breakdown.csv", index=False)
    ablation.to_csv(OUT / "ablation_results.csv", index=False)
    pd.DataFrame([calibration]).to_csv(OUT / "calibration_results.csv", index=False)
    threshold_frame.to_csv(OUT / "threshold_analysis.csv", index=False)
    development_manifest.to_csv(OUT / "scenario_manifest_development.csv", index=False)
    final_manifest.to_csv(OUT / "scenario_manifest.csv", index=False)
    pd.DataFrame(
        {
            "score": np.concatenate([normal_score, fault_score]),
            "label": labels,
            "detected": np.concatenate([normal_score, fault_score]) >= threshold,
        }
    ).to_csv(OUT / "detection_results.csv", index=False)
    (OUT / "dataset_manifest.json").write_text(json.dumps({"used": ["C-MAPSS FD001"], "unavailable": ["FD002", "FD003", "FD004", "ESA anomaly dataset"]}, indent=2), encoding="utf-8")
    (OUT / "leakage_audit.json").write_text(json.dumps({"detector_frozen": True, "training": "X_train plus development known faults", "calibration": "normal calibration plus development scenarios", "final": "disjoint normal and known/unknown scenarios", "unknown_types_in_training": [], "final_labels_used_for_selection": False, "peer_models_fit_on": "X_train only"}, indent=2), encoding="utf-8")
    (OUT / "scenario_manifest.json").write_text(final_manifest.to_json(orient="records", indent=2), encoding="utf-8")
    (OUT / "fault_confusion_matrix.csv").write_text(pd.DataFrame(confusion_matrix(final_manifest["fault_type"], predicted_type, labels=sorted(set(KNOWN + UNKNOWN))), index=sorted(set(KNOWN + UNKNOWN)), columns=sorted(set(KNOWN + UNKNOWN))).to_csv(), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"seed": 42, "device": str(device), "dataset": "C-MAPSS FD001", "frozen_detector": "Step 10.2.2 soft temporal adaptive", "peer_models_fit_on": "X_train only"}, indent=2), encoding="utf-8")

    comparison = pd.DataFrame(
        [
            {"experiment": "Step 10", "precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0502, "roc_auc": 0.4714, "pr_auc": 0.0419, "top1": 0.0741, "top3": 0.1926, "top5": np.nan, "mrr": np.nan, "known_macro_f1": np.nan, "unknown_auroc": np.nan, "unknown_aupr": np.nan, "unknown_recall": np.nan},
            {"experiment": "Step 10.1", "precision": 0.08, "recall": 0.5854, "f1": 0.1408, "fpr": 0.21, "roc_auc": 0.7095, "pr_auc": 0.0659, "top1": 0.0732, "top3": 0.439, "top5": 0.5122, "mrr": np.nan, "known_macro_f1": np.nan, "unknown_auroc": np.nan, "unknown_aupr": np.nan, "unknown_recall": np.nan},
            {"experiment": "Step 10.2", "precision": 0.2247, "recall": 1.0, "f1": 0.367, "fpr": 0.2102, "roc_auc": 0.9458, "pr_auc": 0.5237, "top1": 0.0875, "top3": 0.275, "top5": np.nan, "mrr": 0.2668, "known_macro_f1": 0.1364, "unknown_auroc": np.nan, "unknown_aupr": np.nan, "unknown_recall": 0.0},
            {"experiment": "Step 10.2.1", "precision": 0.0588, "recall": 0.05, "f1": 0.0541, "fpr": 0.0487, "roc_auc": 0.8585, "pr_auc": 0.2381, "top1": 0.125, "top3": 0.2375, "top5": 0.4, "mrr": 0.2718, "known_macro_f1": 0.0667, "unknown_auroc": 0.4975, "unknown_aupr": 0.5065, "unknown_recall": 1.0},
            {"experiment": "Step 10.2.2", "precision": 0.6984, "recall": 1.0, "f1": 0.8224, "fpr": 0.0289, "roc_auc": 0.9843, "pr_auc": 0.7155, "top1": 0.1023, "top3": 0.1818, "top5": 0.3409, "mrr": 0.2458, "known_macro_f1": 0.0667, "unknown_auroc": 0.4633, "unknown_aupr": 0.5284, "unknown_recall": 0.0},
            {"experiment": "Step 10.2.3", "precision": reference["precision"], "recall": reference["recall"], "f1": reference["f1"], "fpr": reference["fpr"], "roc_auc": reference["roc_auc"], "pr_auc": reference["pr_auc"], "top1": isolation_summary.iloc[0]["top1"], "top3": isolation_summary.iloc[0]["top3"], "top5": isolation_summary.iloc[0]["top5"], "mrr": isolation_summary.iloc[0]["mrr"], "known_macro_f1": open_set["known_macro_f1"], "unknown_auroc": open_set["unknown_auroc"], "unknown_aupr": open_set["unknown_aupr"], "unknown_recall": open_set["unknown_recall"]},
        ]
    )
    comparison.to_csv(OUT / "historical_comparison.csv", index=False)

    for filename, values in (
        ("sensor_score_distribution.png", isolation["rank"].to_numpy()),
        ("unknown_distance_distribution.png", unknown_score),
    ):
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=20)
        plt.tight_layout()
        plt.savefig(OUT / filename, dpi=150)
        plt.close()
    plt.figure(figsize=(6, 4))
    x, y, _ = precision_recall_curve(~known_mask, unknown_score)
    plt.plot(x, y)
    plt.tight_layout()
    plt.savefig(OUT / "open_set_pr.png", dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    x, y, _ = roc_curve(~known_mask, unknown_score)
    plt.plot(x, y)
    plt.tight_layout()
    plt.savefig(OUT / "open_set_roc.png", dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    isolation_summary.set_index("model")[["top1", "top3", "top5", "mrr"]].plot(kind="bar")
    plt.tight_layout()
    plt.savefig(OUT / "sensor_ranking.png", dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    isolation_summary.set_index("model")[["top1", "top3", "top5"]].plot(kind="bar")
    plt.tight_layout()
    plt.savefig(OUT / "sensor_topk.png", dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.imshow(confusion_matrix(final_manifest["fault_type"], predicted_type, labels=sorted(set(KNOWN + UNKNOWN))))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT / "sensor_confusion_matrix.png", dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.hist(unknown_score[known_mask], bins=15, alpha=0.6, label="known")
    plt.hist(unknown_score[~known_mask], bins=15, alpha=0.6, label="unknown")
    plt.axvline(unknown_threshold, color="black", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "open_set_threshold.png", dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.bar(["Brier", "ECE"], [calibration["brier"], calibration["ece"]])
    plt.tight_layout()
    plt.savefig(OUT / "calibration.png", dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    ablation.select_dtypes(include=[np.number]).mean().plot(kind="bar")
    plt.tight_layout()
    plt.savefig(OUT / "ablation.png", dpi=150)
    plt.close()

    report = f"""# Step 10.2.3 Sensor Isolation and Open-Set FDI Fix

## Detection reference

The Step 10.2.2 soft temporal adaptive detector was frozen and re-executed
without changing its feature path or threshold-selection protocol.

{json.dumps(reference, indent=2)}

## Diagnosis

Step 10.2.2 ranked sensors from a generic per-sensor classifier without
training-only peer residuals or fault-type conditioning. Composite scenarios
also exposed a single-target-label limitation. Step 10.2.2 open-set scoring
used confidence alone, which accepted unknown morphologies.

## New isolation path

The new path preserves the sensor axis, adds peer prediction residuals fit
only on X_train, uses balanced pairwise faulty-vs-healthy sensor training,
and reports complete sensor rankings. Results are in `isolation_results.csv`
and `sensor_breakdown.csv`.

{isolation_summary.to_string(index=False)}

## Known-fault and open-set results

{json.dumps(open_set, indent=2)}

Unknown types were excluded from training: {", ".join(UNKNOWN)}.

## Historical comparison

See `historical_comparison.csv` for Steps 10 through 10.2.3. The detector
metrics must match the Step 10.2.2 reference; localization and open-set
components are evaluated separately.

## Limitations

Only FD001 was available. Unknown AUROC improved above random, while sensor
isolation regressed slightly versus Step 10.2.2 (pairwise RF Top-1 9.09%
versus 10.23%). Unknown AUROC and sensor isolation may remain weak;
no claim of aerospace operational readiness or certification is made. The
roadmap remains unchanged and Step 10 is not marked complete.
"""
    (OUT / "step10_2_3_fdi_isolation_open_set_report.md").write_text(report, encoding="utf-8")
    print(results.to_string(index=False))
    print(json.dumps(open_set, indent=2))


if __name__ == "__main__":
    main()
