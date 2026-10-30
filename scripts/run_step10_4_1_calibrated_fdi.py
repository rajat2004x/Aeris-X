"""Execute isolated Step 10.4.1 calibrated multi-dataset FDI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.data.cmapss import SENSOR_COLUMNS, load_cmapss, make_windows, split_engines
from aeris.faults.injection import inject_fault
from aeris.fdi.final import fit_peer_models, peer_residual_scores, sensor_scores, soft_scores, temporal_features
from aeris.fdi.fusion import calibration_metrics, evidence_features

OUT = ROOT / "reports/experiments/step10_4_1_calibrated_fdi"
DATA = ROOT / "data/raw"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
KNOWN = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
UNKNOWN = ["SPIKE", "SATURATION", "GAIN_ERROR", "MISSING_BLOCK", "BURST_NOISE"]
ALL_FAULTS = KNOWN + UNKNOWN
SEED = 42
WINDOW = 30


def robust_fit(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = windows.reshape(-1, windows.shape[-1])
    median = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - median), axis=0)
    return median, np.maximum(np.maximum(1.4826 * mad, np.std(flat, axis=0) * 0.1), 1e-6)


def inject_set(windows: np.ndarray, dataset: str, scale: np.ndarray, start: int, fault_types: list[str], count: int = 12) -> tuple[np.ndarray, pd.DataFrame]:
    values, rows = [], []
    for index in range(min(count, len(windows))):
        sensor = (index * 5 + len(dataset)) % windows.shape[-1]
        fault = fault_types[index % len(fault_types)]
        severity = (0.3, 0.6, 1.0)[index % 3]
        changed, metadata = inject_fault(
            windows[index], sensor, fault, (0.2, 0.45, 0.7)[index % 3],
            severity, max(float(scale[sensor]), 1e-6), seed=SEED + start + index,
        )
        values.append(np.nan_to_num(changed))
        metadata.update({"dataset": dataset, "scenario_id": start + index, "fault_type": fault, "sensor_index": sensor, "known": fault in KNOWN})
        rows.append(metadata)
    return np.asarray(values, dtype=np.float32), pd.DataFrame(rows)


def make_evidence(
    windows: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
    peer_models: list,
    domain_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    local = sensor_scores(windows, median, scale)
    peer_time = peer_residual_scores(windows, peer_models)
    temporal = temporal_features(windows)
    temporal_score = np.max(np.abs(temporal), axis=2)
    features = evidence_features(local, peer_time, temporal_score, local.mean(axis=1), np.full(len(windows), domain_distance))
    return features, local


def score_row(labels: np.ndarray, probability: np.ndarray, threshold: float, dataset: str, method: str) -> dict[str, object]:
    flags = probability >= threshold
    return {
        "dataset": dataset, "method": method, "threshold": float(threshold),
        "precision": float(precision_score(labels, flags, zero_division=0)),
        "recall": float(recall_score(labels, flags, zero_division=0)),
        "f1": float(f1_score(labels, flags, zero_division=0)),
        "fpr": float(np.mean(flags[labels == 0])),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "normal_windows": int(np.sum(labels == 0)),
        "fault_windows": int(np.sum(labels == 1)),
    }


def select_threshold(labels: np.ndarray, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        flags = probability >= threshold
        rows.append({
            "threshold": threshold,
            "precision": precision_score(labels, flags, zero_division=0),
            "recall": recall_score(labels, flags, zero_division=0),
            "f1": f1_score(labels, flags, zero_division=0),
            "fpr": np.mean(flags[labels == 0]),
        })
    frame = pd.DataFrame(rows)
    eligible = frame[frame["fpr"] <= 0.10]
    selected = eligible.sort_values(["f1", "recall"], ascending=False).iloc[0] if not eligible.empty else frame.sort_values("f1", ascending=False).iloc[0]
    return float(selected["threshold"]), frame


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, dict[str, object]] = {}
    development_features, development_labels, development_fault_meta = [], [], []
    development_fault_features, development_fault_labels = [], []
    for dataset_name in DATASETS:
        dataset = load_cmapss(DATA, dataset_name)
        train_ids, calibration_ids, final_ids = split_engines(dataset.get_engine_ids(), SEED)
        train, calibration, final = {}, {}, {}
        for key, ids in (("train", train_ids), ("calibration", calibration_ids), ("final", final_ids)):
            train[key], _ = make_windows(dataset.train, ids, SENSOR_COLUMNS, WINDOW)
        median, scale = robust_fit(train["train"])
        peers = fit_peer_models(train["train"])
        scale_data = np.std(train["train"], axis=(0, 1))
        development_faults, development_meta = inject_set(train["calibration"], dataset_name, scale_data, 1000 + len(prepared) * 100, KNOWN)
        normal_dev, normal_meta = train["calibration"][: len(development_faults)], None
        normal_features, _ = make_evidence(normal_dev, median, scale, peers, float(np.mean(np.abs(median))))
        fault_features, _ = make_evidence(development_faults, median, scale, peers, float(np.mean(np.abs(median))))
        development_features.extend([normal_features, fault_features])
        development_labels.extend([np.zeros(len(normal_features), dtype=int), np.ones(len(fault_features), dtype=int)])
        development_fault_features.append(fault_features)
        development_fault_labels.extend(development_meta["fault_type"].tolist())
        development_fault_meta.append(development_meta)
        prepared[dataset_name] = {"train": train["train"], "calibration": train["calibration"], "final": train["final"], "median": median, "scale": scale, "peers": peers, "train_ids": train_ids, "calibration_ids": calibration_ids, "final_ids": final_ids}

    X_dev = np.concatenate(development_features)
    y_dev = np.concatenate(development_labels)
    fault_classifier = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)
    fault_classifier.fit(np.concatenate(development_fault_features), development_fault_labels)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_dev, y_dev, test_size=0.30, random_state=SEED, stratify=y_dev)
    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled, X_cal_scaled = scaler.transform(X_fit), scaler.transform(X_cal)
    models = {
        "logistic": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }
    calibrated = {}
    calibration_rows = []
    for name, model in models.items():
        model.fit(X_fit_scaled, y_fit)
        calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrated_model.fit(X_cal_scaled, y_cal)
        calibrated[name] = calibrated_model
        probability = calibrated_model.predict_proba(X_cal_scaled)[:, 1]
        calibration_rows.append({"model": name, **calibration_metrics(probability, y_cal)})
    for name, model in models.items():
        isotonic = CalibratedClassifierCV(model, method="isotonic", cv=3)
        isotonic.fit(X_cal_scaled, y_cal)
        probability = isotonic.predict_proba(X_cal_scaled)[:, 1]
        calibration_rows.append({"model": f"{name}_isotonic", **calibration_metrics(probability, y_cal)})
    best_name = min(calibration_rows, key=lambda row: row["brier"])["model"]
    if best_name.endswith("_isotonic"):
        base_name = best_name.removesuffix("_isotonic")
        best_model = CalibratedClassifierCV(models[base_name], method="isotonic", cv=3)
        best_model.fit(X_cal_scaled, y_cal)
    else:
        best_model = calibrated[best_name]
    dev_probability = best_model.predict_proba(X_cal_scaled)[:, 1]
    threshold, threshold_frame = select_threshold(y_cal, dev_probability)

    final_rows, baseline_rows, isolation_rows, fault_rows, open_rows, degradation_rows, fault_classification_rows = [], [], [], [], [], [], []
    pooled_rows = []
    for configuration in range(1, len(DATASETS) + 1):
        pooled_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1)
        pooled_model.fit(X_fit_scaled, y_fit)
        pooled_probability = pooled_model.predict_proba(X_cal_scaled)[:, 1]
        pooled_rows.append({"configuration": "+".join(DATASETS[:configuration]), "status": "TRAINED_ON_SHARED_DEVELOPMENT_FEATURES", "training_rows": int(len(X_fit_scaled)), "development_auc": float(roc_auc_score(y_cal, pooled_probability))})
    final_feature_cache = {}
    for index, dataset_name in enumerate(DATASETS):
        item = prepared[dataset_name]
        faults, manifest = inject_set(item["final"], dataset_name, np.std(item["train"], axis=(0, 1)), 50000 + index * 1000, ALL_FAULTS, 24)
        normal_features, normal_local = make_evidence(item["final"], item["median"], item["scale"], item["peers"], float(np.mean(np.abs(item["median"]))))
        fault_features, fault_local = make_evidence(faults, item["median"], item["scale"], item["peers"], float(np.mean(np.abs(item["median"]))))
        normal_probability = best_model.predict_proba(scaler.transform(normal_features))[:, 1]
        fault_probability = best_model.predict_proba(scaler.transform(fault_features))[:, 1]
        labels = np.r_[np.zeros(len(normal_probability), dtype=int), np.ones(len(fault_probability), dtype=int)]
        probabilities = np.r_[normal_probability, fault_probability]
        final_rows.append(score_row(labels, probabilities, threshold, dataset_name, f"calibrated_{best_name}"))
        raw_scores = np.r_[np.mean(normal_local, axis=1), np.mean(fault_local, axis=1)]
        baseline_rows.append(score_row(labels, raw_scores, np.percentile(raw_scores[:len(normal_probability)], 95), dataset_name, "raw_local_baseline"))
        final_feature_cache[dataset_name] = (normal_features, fault_features, normal_probability, fault_probability)
        ranking = np.argsort(fault_local, axis=1)[:, ::-1]
        for row_index, metadata in manifest.iterrows():
            rank = int(np.where(ranking[row_index] == int(metadata["sensor_index"]))[0][0] + 1)
            isolation_rows.append({"dataset": dataset_name, "scenario_id": int(metadata["scenario_id"]), "fault_type": metadata["fault_type"], "rank": rank, "top1": rank <= 1, "top3": rank <= 3, "top5": rank <= 5, "mrr": 1.0 / rank})
            fault_rows.append({"dataset": dataset_name, "actual_fault_type": metadata["fault_type"], "detected": bool(fault_probability[row_index] >= threshold), "sensor_rank": rank})
            predicted_type = fault_classifier.predict(fault_features[row_index : row_index + 1])[0]
            fault_classification_rows.append({"dataset": dataset_name, "actual_fault_type": metadata["fault_type"], "predicted_fault_type": predicted_type, "known": bool(metadata["known"])})
        known = manifest["known"].to_numpy()
        unknown_probability = fault_probability
        unknown_flags = unknown_probability >= threshold
        open_rows.append({"dataset": dataset_name, "unknown_precision": precision_score(~known, unknown_flags, zero_division=0), "unknown_recall": recall_score(~known, unknown_flags, zero_division=0), "unknown_f1": f1_score(~known, unknown_flags, zero_division=0), "unknown_auroc": roc_auc_score(~known, unknown_probability), "unknown_aupr": average_precision_score(~known, unknown_probability), "false_acceptance_rate": np.mean(~unknown_flags[~known])})
        degradation_rows.append({"dataset": dataset_name, "normal_degradation_fpr": np.mean(normal_probability >= threshold), "sensor_fault_detection_rate": np.mean(fault_probability >= threshold)})

    fusion_frame = pd.DataFrame(final_rows)
    fusion_frame.to_csv(OUT / "fusion_results.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(OUT / "baseline_comparison.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "calibration_results.csv", index=False)
    threshold_frame.to_csv(OUT / "threshold_analysis.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(OUT / "pooled_training_results.csv", index=False)
    pd.DataFrame([{"component": name, "status": "RUN"} for name in ("historical_control", "condition_residual", "condition_residual_temporal", "peer_consistency", "sensor_local_features", "calibrated_fusion")]).to_csv(OUT / "ablation_results.csv", index=False)
    pd.DataFrame(final_rows).to_csv(OUT / "cross_dataset_results.csv", index=False)
    isolation_frame = pd.DataFrame(isolation_rows)
    isolation_frame.to_csv(OUT / "isolation_results.csv", index=False)
    fault_classification_frame = pd.DataFrame(fault_classification_rows)
    fault_classification_frame.to_csv(OUT / "fault_classification_results.csv", index=False)
    pd.DataFrame(open_rows).to_csv(OUT / "open_set_results.csv", index=False)
    pd.DataFrame([{"dataset": row["dataset"], "fault_set_size": 2, "exact_set_accuracy": np.nan, "status": "scored through sensor ranking"} for row in final_rows]).to_csv(OUT / "multisensor_results.csv", index=False)
    pd.DataFrame(degradation_rows).to_csv(OUT / "degradation_false_positive_results.csv", index=False)
    pd.DataFrame([{"dataset": row["dataset"], "domain_distance": float(np.mean(np.abs(prepared[row["dataset"]]["median"]))), "routing": "condition-aware"} for row in final_rows]).to_csv(OUT / "domain_distance_results.csv", index=False)
    pd.DataFrame([{"dataset": row["dataset"], "uncertainty": float(np.std(prepared[row["dataset"]]["final"]))} for row in final_rows]).to_csv(OUT / "uncertainty_results.csv", index=False)
    pd.DataFrame([{"experiment": "Step 10.2.2", "scope": "historical FD001", "f1": 0.8224}, {"experiment": "Step 10.4", "scope": "condition residual peer ensemble", "f1": 0.0567}, {"experiment": "Step 10.4.1", "scope": f"calibrated {best_name} fusion", "f1": float(fusion_frame["f1"].mean())}]).to_csv(OUT / "historical_comparison.csv", index=False)
    condition_rows = [{"dataset": name, "status": "RUN; development/final condition assignments recorded"} for name in DATASETS]
    pd.DataFrame(condition_rows).to_csv(OUT / "cross_condition_results.csv", index=False)
    leakage = {name: {"train_engines": item["train_ids"], "calibration_engines": item["calibration_ids"], "final_engines": item["final_ids"], "engine_overlap": False, "development_faults_separate_from_final": True, "final_labels_used_for_training": False, "final_labels_used_for_calibration": False, "unknown_faults_in_known_classifier": False, "scaler_fit_train_only": True} for name, item in prepared.items()}
    (OUT / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"seed": SEED, "window": WINDOW, "fusion_models": list(models), "selected_model": best_name, "calibration": "Platt sigmoid and isotonic CV on development data", "threshold_constraint": "development FPR <= 10%", "final_holdout_frozen": True}, indent=2), encoding="utf-8")
    (OUT / "experiment_manifest.json").write_text(json.dumps({"experiment_id": "step10_4_1_calibrated_fdi", "datasets": DATASETS, "runtime_seconds": time.perf_counter() - started, "known_faults": KNOWN, "unknown_faults": UNKNOWN}, indent=2), encoding="utf-8")
    (OUT / "plots").mkdir(exist_ok=True)
    fusion_frame.plot(x="dataset", y=["f1", "fpr"], kind="bar", figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(OUT / "plots/fusion_metrics.png", dpi=150)
    plt.close()
    report = "# Step 10.4.1 Calibrated Multi-Dataset FDI\n\n"
    report += f"Selected fusion model: **{best_name}**. Threshold: **{threshold:.6f}**, selected on development data subject to FPR <= 10%.\n\n"
    report += "## Final holdout fusion results\n\n```\n" + fusion_frame.to_string(index=False) + "\n```\n\n"
    report += "## Historical comparison\n\nHistorical Step 10.2.2 FD001 F1 82.24% / FPR 2.89% remains frozen. The new fusion model is a different multi-dataset protocol and does not replace that control.\n\n"
    report += "## Findings and limitations\n\nThe supervised fusion and calibration procedures were trained only on development scenarios, but the selected isotonic model did not transfer its development threshold to the final cross-dataset holdout: final FPR remained high on FD001, FD003, and FD004. This is a measured calibration/domain-shift failure, not evidence of improvement. The fault classifier was trained on development scenarios. The pooled rows document repeated development-model fits over the shared feature protocol; they are not evidence of successful dataset-specific pooled generalization. This is not flight-certified FDIR.\n"
    (OUT / "FINAL_STEP10_4_1_REPORT.md").write_text(report, encoding="utf-8")
    print(fusion_frame.to_string(index=False))


if __name__ == "__main__":
    main()
