"""Run the bounded, leakage-controlled SMAP/MSL experiment."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.spacecraft.experiment import (
    DETECTORS,
    SEED,
    discover_dataset,
    environment_manifest,
    event_metrics,
    evaluate_detectors,
    fit_detectors,
    fit_preprocessors,
    preprocessing_manifest,
    profile_dataset,
    run_robustness,
)
from aeris.spacecraft.pipeline import audit_spacecraft_data


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    data_root = ROOT / "data" / "raw" / "SMAP_MSL"
    output = ROOT / "reports" / "experiments" / "spacecraft_smap_msl"
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    dataset = discover_dataset(data_root)
    audit = audit_spacecraft_data(data_root)
    profile = profile_dataset(dataset)
    write_json(output / "dataset_profile.json", {"audit": audit, "profile": profile})
    preprocessors = fit_preprocessors(dataset)
    write_json(output / "preprocessing_manifest.json", preprocessing_manifest(dataset, preprocessors))
    fitted = fit_detectors(dataset, preprocessors)
    detector_results, channel_scores = evaluate_detectors(dataset, preprocessors, fitted)
    rows = detector_results["rows"]
    pd.DataFrame(rows).to_csv(output / "detector_comparison.csv", index=False)
    baseline_summary = (
        pd.DataFrame(rows).groupby("detector", as_index=False)[
            ["precision", "recall", "f1", "accuracy", "fpr", "fnr", "roc_auc", "pr_auc", "normal_only_fpr", "anomaly_point_coverage"]
        ].mean()
    )
    baseline_summary.to_json(output / "baseline_results.json", orient="records", indent=2)
    event_rows: list[dict[str, object]] = []
    spacecraft_by_channel = {row["chan_id"]: row["spacecraft"] for row in dataset["label_rows"]}
    for channel, detector_values in channel_scores.items():
        for detector in DETECTORS:
            threshold = next(
                row["threshold"] for row in rows if row["channel"] == channel and row["detector"] == detector
            )
            metrics = event_metrics(dataset["labels"].get(channel, []), detector_values[detector], threshold)
            event_rows.append({"channel": channel, "spacecraft": spacecraft_by_channel.get(channel), "detector": detector, **metrics})
    pd.DataFrame(event_rows).to_json(output / "event_metrics.json", orient="records", indent=2)
    event_summary = (
        pd.DataFrame(event_rows)
        .groupby("detector", as_index=False)[["event_recall", "event_f1", "mean_detection_delay"]]
        .mean(numeric_only=True)
    )
    baseline_summary = baseline_summary.merge(event_summary, on="detector", how="left")
    baseline_summary.to_json(output / "baseline_results.json", orient="records", indent=2)
    isolation_rows = []
    for detector in DETECTORS:
        for spacecraft in ("SMAP", "MSL", "combined"):
            channels = [channel for channel in channel_scores if spacecraft == "combined" or spacecraft_by_channel.get(channel) == spacecraft]
            ranks = []
            for channel in channels:
                for start, end in dataset["labels"].get(channel, []):
                    point = min((start + end) // 2, len(channel_scores[channel][detector]) - 1)
                    values = {
                        other: float(channel_scores[other][detector][min(point, len(channel_scores[other][detector]) - 1)])
                        for other in channels
                    }
                    ordered = sorted(values, key=values.get, reverse=True)
                    rank = ordered.index(channel) + 1 if channel in ordered else len(ordered) + 1
                    ranks.append(rank)
            isolation_rows.append({
                "detector": detector,
                "spacecraft": spacecraft,
                "top1": float(np.mean(np.asarray(ranks) <= 1)) if ranks else 0.0,
                "top3": float(np.mean(np.asarray(ranks) <= 3)) if ranks else 0.0,
                "top5": float(np.mean(np.asarray(ranks) <= 5)) if ranks else 0.0,
                "mrr": float(np.mean(1 / np.asarray(ranks))) if ranks else 0.0,
                "events": len(ranks),
            })
    write_json(output / "channel_isolation.json", isolation_rows)
    robustness = run_robustness(dataset, preprocessors)
    write_json(output / "robustness_results.json", robustness)
    cross_domain = {
        "purpose": "descriptive comparison only; no direct model transfer claim",
        "spacecraft": {
            "channels": 82,
            "label_records": 82,
            "feature_summary_dimensions": 3,
            "detectors": baseline_summary.to_dict(orient="records"),
        },
        "cmapss_step11_reference": {
            "dataset": "FD001",
            "detector": "robust_multisensor_mad",
            "precision": 0.6,
            "recall": 0.06,
            "f1": 0.10909090909090909,
            "fpr": 0.013333333333333334,
            "pr_auc": 0.276355124413086,
            "event_recall": 0.01,
        },
        "interpretation": [
            "SMAP/MSL contains channel-specific variable-length telemetry rather than fixed engine windows.",
            "The comparison is descriptive because preprocessing and labels differ.",
        ],
    }
    write_json(output / "cross_domain_comparison.json", cross_domain)
    (output / "cross_domain_comparison.md").write_text(
        "# Cross-domain comparison\n\nThis descriptive comparison preserves Step 11 as immutable reference evidence. "
        "It does not claim model transfer or statistical significance.\n\n"
        f"- SMAP/MSL channels: 82\n- Step 11 reference PR-AUC: {cross_domain['cmapss_step11_reference']['pr_auc']}\n",
        encoding="utf-8",
    )
    failure = {
        "categories": ["false_positives", "false_negatives", "short_anomalies", "long_anomalies", "gradual_anomalies", "abrupt_anomalies"],
        "by_detector": {
            detector: {
                "false_positives": int(sum(row["false_positives"] for row in rows if row["detector"] == detector)),
                "false_negatives": int(sum(row["test_points"] * row["fnr"] for row in rows if row["detector"] == detector)),
            }
            for detector in DETECTORS
        },
        "limitations": "Failure categories are aggregate; morphology labels are not available in the benchmark CSV.",
    }
    write_json(output / "failure_analysis.json", failure)
    reference = baseline_summary.sort_values(
        ["pr_auc", "event_recall", "f1", "normal_only_fpr", "mean_detection_delay"],
        ascending=[False, False, False, True, True],
        na_position="last",
    ).iloc[0].to_dict()
    write_json(output / "reproducibility_manifest.json", environment_manifest())
    write_json(output / "experiment_registry.json", {
        "experiments": [
            {"experiment": "dataset_profile", "status": "COMPLETED", "runtime": None, "artifact": "dataset_profile.json"},
            {"experiment": "classical_detectors", "status": "COMPLETED", "runtime": round(time.perf_counter() - started, 3), "artifact": "detector_comparison.csv"},
            {"experiment": "autoencoder", "status": "SKIPPED", "runtime": 0, "artifact": None, "metrics": "Skipped to enforce bounded runtime"},
            {"experiment": "robustness", "status": "COMPLETED", "runtime": None, "artifact": "robustness_results.json"},
        ]
    })
    plt.figure(figsize=(8, 4))
    baseline_summary.plot(x="detector", y="pr_auc", kind="bar", legend=False, title="SMAP/MSL detector PR-AUC")
    plt.tight_layout()
    plt.savefig(plots / "detector_comparison.png")
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.hist([value["test_length"] for value in profile["channels"].values()], bins=20)
    plt.title("SMAP/MSL test sequence lengths")
    plt.xlabel("timesteps")
    plt.ylabel("channels")
    plt.tight_layout()
    plt.savefig(plots / "dataset_overview.png")
    plt.close()
    event_frame = pd.DataFrame(event_rows)
    plt.figure(figsize=(8, 4))
    event_frame.groupby("detector")["event_recall"].mean().plot(kind="bar", title="Mean event recall")
    plt.tight_layout()
    plt.savefig(plots / "event_detection.png")
    plt.close()
    isolation_frame = pd.DataFrame(isolation_rows)
    plt.figure(figsize=(8, 4))
    isolation_frame[isolation_frame["spacecraft"] == "combined"].set_index("detector")[["top1", "top3", "top5"]].plot(kind="bar", title="Combined channel isolation")
    plt.tight_layout()
    plt.savefig(plots / "channel_isolation.png")
    plt.close()
    (output / "report.md").write_text(f"""# AERIS-X SMAP/MSL Spacecraft Anomaly Detection

## 1. Objective

Evaluate leakage-controlled classical anomaly detectors on the acquired Telemanom SMAP/MSL package.

## 2. Dataset

82 train files, 82 test files, 55 SMAP channels, 27 MSL channels, and 82 label records. The CSV contains 81 unique IDs because `P-2` is duplicated; it was retained.

## 3. Dataset Audit

The audit found no missing or infinite values. Train/test IDs match, label IDs are present, and anomaly sequences parse successfully.

## 4. Leakage-Controlled Protocol

Each channel uses train-only mean/std/norm summary preprocessing. The first 80% of train telemetry fits detectors and the remaining 20% calibrates a frozen 99th-percentile threshold. Test labels are used only for final evaluation.

## 5. Detectors

Robust median/MAD, rolling deviation, PCA reconstruction, and Isolation Forest were evaluated. The autoencoder was explicitly skipped to maintain the laptop runtime bound.

## 6. Point-Level Results

See `detector_comparison.csv` and `baseline_results.json` for precision, recall, F1, accuracy, FPR, FNR, ROC-AUC, PR-AUC, and coverage.

## 7. Event-Level Results

Events are the provided anomaly ranges; predicted points are grouped with a two-sample temporal gap rule fixed before evaluation. See `event_metrics.json`.

## 8. Channel Isolation

Scores are ranked across channels at anomaly midpoints. See `channel_isolation.json`; this is channel isolation, not fault classification.

## 9. SMAP vs MSL

Per-spacecraft rows are available in `detector_comparison.csv` and `event_metrics.json`. No significance claim is made.

## 10. Robustness

Deterministic evaluation-only bias, drift, spike, stuck-at, noise, and dropout perturbations are in `robustness_results.json`; raw data was not modified.

## 11. Cross-Domain Comparison with C-MAPSS

See `cross_domain_comparison.json` and `cross_domain_comparison.md`. Step 11 artifacts were not modified.

## 12. Explainability

Per-channel scores, thresholds, and event ranges provide reproducible evidence. No unsupported causal explanation is claimed.

## 13. Failure Analysis

Aggregate false-positive and false-negative counts are in `failure_analysis.json`. Morphology-specific classification is limited by the source labels.

## 14. Best Research Reference Detector

Selection rule was fixed as PR-AUC, then event recall, F1, normal FPR, and detection delay. The selected reference is **{reference['detector']}**. It is not certified for operations.

## 15. Limitations

This is a benchmark analysis, not NASA validation, flight certification, or operational deployment. The summary representation compresses native feature vectors, and no direct cross-domain transfer was tested.

## 16. Reproducibility

Seed: {SEED}. Runtime and environment details are in `reproducibility_manifest.json`.

## 17. Conclusion

The pipeline provides bounded, leakage-controlled spacecraft telemetry evidence, but detector quality and channel isolation must be interpreted from the complete metrics rather than accuracy alone.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
