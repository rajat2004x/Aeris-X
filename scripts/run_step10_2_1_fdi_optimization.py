"""Run the isolated Step 10.2.1 precision and isolation optimization."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.data.adapters.cmapss_adapter import load_prepared_cmapss
from aeris.evaluation.anomaly import fit_robust_baseline
from aeris.faults.injection import inject_fault
from aeris.fdi.optimization import persistence_flags, sensor_feature_tensor, sensor_set_metrics
from aeris.training.trainer import seed_everything, select_device


OUTPUT = PROJECT_ROOT / "reports/experiments/step10_2_1_fdi_optimization"
KNOWN = ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"]
UNKNOWN = ["SPIKE", "SATURATION", "GAIN_ERROR", "MISSING_BLOCK", "BURST_NOISE"]


def make_scenarios(
    sources: np.ndarray,
    fault_types: list[str],
    stds: np.ndarray,
    start: int,
    composite: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    windows = []
    records: list[dict[str, Any]] = []
    scenario = start
    for fault_type in fault_types:
        for source_index in range(min(8, len(sources))):
            target = (source_index * 3 + len(fault_type)) % sources.shape[2]
            targets = [target]
            modified, metadata = inject_fault(
                sources[source_index],
                target,
                fault_type,
                (source_index % 3 + 1) / 4,
                (source_index % 3 + 1) / 3,
                max(float(stds[target]), 1e-6),
                seed=42 + scenario,
            )
            if composite and source_index % 4 == 0:
                second = (target + 1) % sources.shape[2]
                modified, _ = inject_fault(
                    modified,
                    second,
                    "BIAS",
                    0.5,
                    0.5,
                    max(float(stds[second]), 1e-6),
                    seed=1000 + scenario,
                )
                targets.append(second)
            windows.append(np.nan_to_num(modified, nan=0.0, posinf=0.0, neginf=0.0))
            metadata.update(
                {
                    "scenario_index": scenario,
                    "source_index": source_index,
                    "fault_type": fault_type,
                    "target_sensors": json.dumps(targets),
                    "known_fault": fault_type in KNOWN,
                    "split": "development" if start == 0 else "final_test",
                    "severity_name": ["low", "medium", "high"][source_index % 3],
                }
            )
            records.append(metadata)
            scenario += 1
    return np.asarray(windows, dtype=np.float32), pd.DataFrame(records)


def alarm_score(windows: np.ndarray, median: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.nan_to_num(windows)
    z = np.abs((values - median[None, None, :]) / np.maximum(scale[None, None, :], 1e-6))
    sensor_scores = np.max(z, axis=1)
    persistent = np.zeros((len(values), values.shape[2]))
    for sample in range(len(values)):
        flags = z[sample] >= 3.0
        for sensor in range(values.shape[2]):
            run = best = 0
            for flag in flags[:, sensor]:
                run = run + 1 if flag else 0
                best = max(best, run)
            persistent[sample, sensor] = best
    score = np.max(sensor_scores * np.maximum(persistent / 3.0, 1 / 3), axis=1)
    return score, sensor_scores


def choose_threshold(normal: np.ndarray, faults: np.ndarray) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(np.percentile(np.concatenate([normal, faults]), np.linspace(85, 99.5, 60)))
    rows = []
    for threshold in candidates:
        labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(faults), dtype=int)])
        scores = np.concatenate([normal, faults])
        flags = scores >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(labels, flags, zero_division=0)),
                "recall": float(recall_score(labels, flags, zero_division=0)),
                "f1": float(f1_score(labels, flags, zero_division=0)),
                "fpr": float(np.mean(flags[: len(normal)])),
            }
        )
    frame = pd.DataFrame(rows)
    eligible = frame[frame["fpr"] <= 0.10]
    selected = eligible.sort_values(["f1", "recall"], ascending=False).iloc[0] if not eligible.empty else frame.iloc[0]
    return float(selected["threshold"]), frame


def isolation_report(
    classifier: Any,
    features: np.ndarray,
    scenarios: pd.DataFrame,
    name: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    flattened = features.reshape(len(features), features.shape[1], -1)
    rows = []
    predicted_sets = []
    actual_sets = []
    for index, (_, scenario) in enumerate(scenarios.iterrows()):
        probabilities = classifier.predict_proba(flattened[index])
        ranking = np.argsort(probabilities.max(axis=1))[::-1]
        target = set(json.loads(scenario["target_sensors"]))
        guess = set(ranking[: len(target)])
        predicted_sets.append(guess)
        actual_sets.append(target)
        rank = min(int(np.where(ranking == value)[0][0] + 1) for value in target)
        rows.append(
            {
                "classifier": name,
                "scenario_index": int(scenario["scenario_index"]),
                "fault_type": scenario["fault_type"],
                "rank": rank,
                "top1": rank <= 1,
                "top3": rank <= 3,
                "top5": rank <= 5,
                "mrr": 1.0 / rank,
                "actual_sensors": json.dumps(sorted(int(value) for value in target)),
                "predicted_sensors": json.dumps(sorted(int(value) for value in guess)),
            }
        )
    result = pd.DataFrame(rows)
    metrics = {
        "top1": float(result["top1"].mean()),
        "top3": float(result["top3"].mean()),
        "top5": float(result["top5"].mean()),
        "mrr": float(result["mrr"].mean()),
        "mean_rank": float(result["rank"].mean()),
        **sensor_set_metrics(predicted_sets, actual_sets),
    }
    return result, metrics


def main() -> None:
    seed_everything(42)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_prepared_cmapss(PROJECT_ROOT / "data/processed/cmapss_fd001")
    X_train = np.nan_to_num(data["X_train"]).astype(np.float32)
    X_validation = np.nan_to_num(data["X_validation"]).astype(np.float32)
    calibration, final_normal = np.array_split(X_validation, 2)
    baseline = fit_robust_baseline(X_train)
    stds = np.std(X_train, axis=(0, 1))
    development, development_manifest = make_scenarios(calibration, KNOWN, stds, 0, composite=True)
    final, final_manifest = make_scenarios(final_normal, KNOWN + UNKNOWN, stds, len(development_manifest), composite=True)

    normal_calibration_score, _ = alarm_score(calibration, baseline["median"], baseline["scale"])
    development_score, development_sensor = alarm_score(development, baseline["median"], baseline["scale"])
    normal_final_score, normal_final_sensor = alarm_score(final_normal, baseline["median"], baseline["scale"])
    final_score, final_sensor = alarm_score(final, baseline["median"], baseline["scale"])
    threshold, threshold_frame = choose_threshold(normal_calibration_score, development_score)

    labels = np.concatenate([np.zeros(len(final_normal), dtype=int), np.ones(len(final), dtype=int)])
    scores = np.concatenate([normal_final_score, final_score])
    flags = scores >= threshold
    results = {
        "detector": "persistent_adaptive_statistical",
        "threshold": threshold,
        "precision": float(precision_score(labels, flags, zero_division=0)),
        "recall": float(recall_score(labels, flags, zero_division=0)),
        "f1": float(f1_score(labels, flags, zero_division=0)),
        "fpr": float(np.mean(flags[: len(final_normal)])),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "detection_rate": float(np.mean(flags[len(final_normal) :])),
    }

    train_features = sensor_feature_tensor(development, baseline["median"], baseline["scale"])
    test_features = sensor_feature_tensor(final, baseline["median"], baseline["scale"])
    train_flat = train_features.reshape(len(development) * train_features.shape[1], -1)
    train_labels = np.concatenate(
        [np.full(train_features.shape[1], target, dtype=int) for target in development_manifest["target_sensors"].map(lambda value: json.loads(value)[0])]
    )
    test_flat = test_features.reshape(len(final) * test_features.shape[1], -1)
    classifiers = {
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=2),
    }
    isolation_frames = []
    classifier_metrics: dict[str, dict[str, float]] = {}
    for name, classifier in classifiers.items():
        classifier.fit(train_flat, train_labels)
        joblib.dump(classifier, OUTPUT / f"{name}_sensor_classifier.joblib")
        frame, metrics = isolation_report(classifier, test_features, final_manifest, name)
        isolation_frames.append(frame)
        classifier_metrics[name] = metrics
        (OUTPUT / f"{name}_sensor_classifier.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    isolation = pd.concat(isolation_frames, ignore_index=True)

    fault_train_features = sensor_feature_tensor(development, baseline["median"], baseline["scale"]).mean(axis=1)
    fault_test_features = sensor_feature_tensor(final, baseline["median"], baseline["scale"]).mean(axis=1)
    fault_classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    fault_classifier.fit(fault_train_features, development_manifest["fault_type"])
    joblib.dump(fault_classifier, OUTPUT / "fault_type_classifier.joblib")
    probabilities = fault_classifier.predict_proba(fault_test_features)
    predicted = fault_classifier.classes_[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    rejection_threshold = float(np.percentile(
        fault_classifier.predict_proba(fault_train_features).max(axis=1), 10
    ))
    rejected = confidence < rejection_threshold
    classification = pd.DataFrame(
        {
            "scenario_index": final_manifest["scenario_index"],
            "actual_fault_type": final_manifest["fault_type"],
            "predicted_fault_type": predicted,
            "confidence": confidence,
            "rejected_unknown": rejected,
            "known_fault": final_manifest["known_fault"],
            "severity_name": final_manifest["severity_name"],
        }
    )
    known = classification["known_fault"].to_numpy()
    known_mask = known
    classification_metrics = {
        "known_macro_f1": float(
            f1_score(
                classification.loc[known_mask, "actual_fault_type"],
                classification.loc[known_mask, "predicted_fault_type"],
                average="macro",
                zero_division=0,
            )
        ),
        "unknown_recall": float(np.mean(rejected[~known])),
        "unknown_precision": float(np.sum(rejected & ~known) / max(1, np.sum(rejected))),
        "unknown_auroc": float(roc_auc_score(~known, 1 - confidence)),
        "unknown_aupr": float(average_precision_score(~known, 1 - confidence)),
        "false_acceptance_rate": float(np.mean(~rejected[~known])),
    }
    confusion = confusion_matrix(
        classification["actual_fault_type"],
        classification["predicted_fault_type"],
        labels=sorted(set(KNOWN + UNKNOWN)),
    )

    pd.DataFrame([results]).to_csv(OUTPUT / "results.csv", index=False)
    pd.DataFrame([results]).to_json(OUTPUT / "results.json", orient="records", indent=2)
    isolation.to_csv(OUTPUT / "sensor_isolation.csv", index=False)
    classification.to_csv(OUTPUT / "fault_classification.csv", index=False)
    pd.DataFrame([classification_metrics]).to_csv(OUTPUT / "open_set_results.csv", index=False)
    pd.DataFrame(
        confusion,
        index=sorted(set(KNOWN + UNKNOWN)),
        columns=sorted(set(KNOWN + UNKNOWN)),
    ).to_csv(OUTPUT / "fault_confusion_matrix.csv")
    threshold_frame.to_csv(OUTPUT / "threshold_analysis.csv", index=False)
    error_rows = []
    for fault_type, group in classification.groupby("actual_fault_type"):
        scores_group = group["rejected_unknown"].to_numpy()
        error_rows.append(
            {
                "fault_type": fault_type,
                "sample_count": len(group),
                "classification_accuracy": float(np.mean(group["actual_fault_type"] == group["predicted_fault_type"])),
                "unknown_rejection_rate": float(np.mean(scores_group)),
                "severity": group["severity_name"].iloc[0],
            }
        )
    pd.DataFrame(error_rows).to_csv(OUTPUT / "error_analysis.csv", index=False)
    pd.DataFrame([{"model": name, **metrics} for name, metrics in classifier_metrics.items()]).to_csv(
        OUTPUT / "ablation_results.csv", index=False
    )
    best_isolation = max(classifier_metrics.values(), key=lambda metrics: metrics["top1"])
    comparison = pd.DataFrame(
        [
            {
                "experiment": "Step 10",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "pr_auc": 0.0419,
                "roc_auc": 0.4714,
                "fpr": 0.0502,
                "top1": 0.0741,
                "top3": 0.1926,
                "mrr": np.nan,
                "fault_macro_f1": np.nan,
                "unknown_recall": np.nan,
            },
            {
                "experiment": "Step 10.1",
                "precision": 0.0800,
                "recall": 0.5854,
                "f1": 0.1408,
                "pr_auc": 0.0659,
                "roc_auc": 0.7095,
                "fpr": 0.2100,
                "top1": 0.0732,
                "top3": 0.4390,
                "mrr": np.nan,
                "fault_macro_f1": np.nan,
                "unknown_recall": np.nan,
            },
            {
                "experiment": "Step 10.2",
                "precision": 0.2247,
                "recall": 1.0,
                "f1": 0.3670,
                "pr_auc": 0.5237,
                "roc_auc": 0.9458,
                "fpr": 0.2102,
                "top1": 0.0875,
                "top3": 0.2750,
                "mrr": 0.2668,
                "fault_macro_f1": 0.1364,
                "unknown_recall": 0.0,
            },
            {
                "experiment": "Step 10.2.1",
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
                "pr_auc": results["pr_auc"],
                "roc_auc": results["roc_auc"],
                "fpr": results["fpr"],
                "top1": best_isolation["top1"],
                "top3": best_isolation["top3"],
                "mrr": best_isolation["mrr"],
                "fault_macro_f1": classification_metrics["known_macro_f1"],
                "unknown_recall": classification_metrics["unknown_recall"],
            },
        ]
    )
    comparison.to_csv(OUTPUT / "comparison_matrix.csv", index=False)
    development_manifest.to_csv(OUTPUT / "scenario_manifest_development.csv", index=False)
    final_manifest.to_csv(OUTPUT / "scenario_manifest.csv", index=False)
    (OUTPUT / "leakage_audit.json").write_text(
        json.dumps(
            {
                "training": "FD001 calibration-derived known-fault scenarios",
                "threshold_calibration": "normal calibration plus development known faults only",
                "final_test": "FD001 final normal windows plus disjoint known/unknown scenarios",
                "scenario_overlap": False,
                "unknown_faults_excluded_from_training": UNKNOWN,
                "final_labels_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "used": ["C-MAPSS FD001"],
                "unavailable": ["FD002", "FD003", "FD004", "ESA anomaly dataset"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, values in (("pr_curve", (labels, scores)), ("roc_curve", (labels, scores))):
        plt.figure(figsize=(6, 4))
        if name == "pr_curve":
            x, y, _ = precision_recall_curve(*values)
            plt.plot(x, y)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
        else:
            x, y, _ = roc_curve(*values)
            plt.plot(x, y)
            plt.xlabel("False-positive rate")
            plt.ylabel("Recall")
        plt.tight_layout()
        plt.savefig(OUTPUT / f"{name}.png", dpi=150)
        plt.close()
    report = f"""# Step 10.2.1 FDI Precision, Isolation, and Open-Set Optimization

## Decision

This is a new isolated optimization experiment. Step 10 remains unmarked and
the roadmap remains unchanged. Only C-MAPSS FD001 was available; FD002-FD004
and ESA telemetry were not used.

## Why the previous system failed

Step 10.2 used attribution from global detector scores as a proxy for sensor
isolation, trained fault classification on aggregate features, and had no
explicit unknown rejection. Its 21% FPR and 0% unknown recall therefore did
not establish reliable FDI.

## Final locked detection result

{json.dumps(results, indent=2)}

## Comparison matrix

See `comparison_matrix.csv` for the Step 10 through Step 10.2.1 comparison.
The selected detector prioritizes F1, then PR-AUC, then low FPR; isolation
and unknown rejection are reported separately rather than hidden in a single
aggregate score.

Threshold selection used normal calibration and development known-fault data
only, with a persistence-aware score and an FPR objective of at most 10%.

## Dedicated sensor classifiers

{json.dumps(classifier_metrics, indent=2)}

The sensor classifiers consume one feature vector per sensor and are evaluated
on disjoint final scenarios. Detailed per-scenario ranks and sensor sets are
in `sensor_isolation.csv`.

## Fault classification and open-set rejection

{json.dumps(classification_metrics, indent=2)}

Unknown classes were excluded from training: {", ".join(UNKNOWN)}. The
rejection rule is confidence-based and its threshold is derived from the
training-known confidence distribution. Results are reported even if weak.

## Limitations

This remains a synthetic-fault FD001 study. Cross-condition and cross-dataset
generalization cannot be measured without the unavailable datasets. The
classifier does not establish physical causality or operational aerospace
readiness. Historical Step 10, Step 10.1, and Step 10.2 artifacts were not
overwritten.
"""
    (OUTPUT / "optimization_report.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame([results]).to_string(index=False))
    print(json.dumps(classification_metrics, indent=2))


if __name__ == "__main__":
    main()
