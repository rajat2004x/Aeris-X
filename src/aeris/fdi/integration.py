"""Assemble the defensible final Step 10 FDI evidence without retraining."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


FINAL_SOURCE = Path("reports/experiments/step10_4_final_fdi")
HISTORICAL_SOURCE = Path("reports/experiments/step10_4_3_multiobjective_fdi")


def _read_csv(source: Path, name: str) -> pd.DataFrame:
    path = source / name
    if not path.exists():
        raise FileNotFoundError(f"Required Step 10 evidence is missing: {path}")
    return pd.read_csv(path)


def _copy_evidence(source: Path, name: str, output: Path) -> None:
    path = source / name
    if not path.exists():
        raise FileNotFoundError(f"Required Step 10 evidence is missing: {path}")
    shutil.copyfile(path, output / name)


def _classification_summary(source: Path) -> pd.DataFrame:
    values = _read_csv(source, "fault_classification_results.csv")
    if not {"actual_fault_type", "predicted_fault_type"} <= set(values.columns):
        return pd.DataFrame({
            "dataset": sorted(values["dataset"].unique()),
            "known_scenarios": [None] * values["dataset"].nunique(),
            "exact_fault_type_accuracy": [None] * values["dataset"].nunique(),
            "status": ["not_evaluable_from_final_artifact"] * values["dataset"].nunique(),
        })
    known = values if "known" not in values.columns else values[values["known"].astype(bool)]
    return (
        known.groupby("dataset")
        .apply(
            lambda group: pd.Series({
                "known_scenarios": len(group),
                "exact_fault_type_accuracy": float(
                    (group["actual_fault_type"] == group["predicted_fault_type"]).mean()
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )


def _write_plots(detection: pd.DataFrame, isolation: pd.DataFrame, output: Path) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    detection.set_index("dataset")[["precision", "recall", "f1", "fpr"]].plot.bar(ax=axis)
    axis.set_title("Final integrated detection metrics by dataset")
    axis.set_ylabel("Rate")
    figure.tight_layout()
    figure.savefig(plots / "detection_by_dataset.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    isolation.groupby("dataset")[["top1", "top3", "top5", "top10"]].mean().plot.bar(ax=axis)
    axis.set_title("Final integrated sensor isolation by dataset")
    axis.set_ylabel("Hit rate")
    figure.tight_layout()
    figure.savefig(plots / "isolation_by_dataset.png", dpi=150)
    plt.close(figure)


def _write_report(
    detection: pd.DataFrame,
    isolation: pd.DataFrame,
    classification: pd.DataFrame,
    open_set: pd.DataFrame,
    calibration: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Final Step 10: Sensor Health and FDI",
        "",
        "## Executive summary",
        "This isolated milestone artifact packages the defensible condition-aware FDI evidence already produced by the Step 10.4 final experiment. No model was retrained and no historical Step 10.x result was overwritten.",
        "The integrated framework provides validation, condition-residual and temporal-peer evidence, thresholded detection, sensor ranking, fault-type evidence, open-set summaries, and calibration/uncertainty records.",
        "",
        "## Research question",
        "Can AERIS-X detect abnormal sensor behavior, distinguish it from normal degradation, identify likely sensors, and report confidence across operating conditions?",
        "",
        "## Final integrated architecture",
        "Telemetry -> validation -> condition residuals -> temporal evidence and peer residuals -> normal-only threshold -> FDI decision -> sensor ranking -> fault/open-set evidence -> evaluation.",
        "The selected detector is the historical condition-residual + soft temporal evidence + peer-consistency ensemble. The final artifact is an evidence integration layer, not a new training run.",
        "",
        "## Final integrated detection results",
        "",
        "| Dataset | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detection.itertuples():
        lines.append(
            f"| {row.dataset} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | "
            f"{row.fpr:.4f} | {row.roc_auc:.4f} | {row.pr_auc:.4f} |"
        )
    lines.extend([
        "",
        "## Sensor isolation",
        "",
        "| Dataset | Top-1 | Top-3 | Top-5 | Top-10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in isolation.groupby("dataset")[["top1", "top3", "top5", "top10", "mrr"]].mean().reset_index().itertuples():
        lines.append(f"| {row.dataset} | {row.top1:.4f} | {row.top3:.4f} | {row.top5:.4f} | {row.top10:.4f} | {row.mrr:.4f} |")
    lines.extend([
        "",
        "## Fault classification",
        "Classification is evidence only; it is not presented as a validated universal classifier.",
        "",
        "| Dataset | Known scenarios | Exact fault-type accuracy |",
        "|---|---:|---:|",
    ])
    for row in classification.itertuples():
        count = "N/A" if row.known_scenarios is None else str(int(row.known_scenarios))
        accuracy = "N/A" if row.exact_fault_type_accuracy is None else f"{row.exact_fault_type_accuracy:.4f}"
        lines.append(f"| {row.dataset} | {count} | {accuracy} |")
    lines.extend([
        "",
        "## Open-set and calibration evidence",
        "",
        "```text",
        open_set.to_string(index=False),
        "```",
        "",
        "```text",
        calibration.to_string(index=False),
        "```",
        "",
        "## Historical findings and limitations",
        "- The selected final artifact contains sensor-fault detection rows but no actual-versus-predicted fault-type fields; fault-type classification accuracy is therefore not claimed.",
        "- FD001–FD004 results are not interchangeable with the strong historical FD001-only result from Step 10.2.2.",
        "- The final integrated detector has very low precision and F1 on all four datasets despite useful ranking metrics in some datasets.",
        "- Normal degradation and sensor anomalies remain difficult to separate; a changing telemetry channel is not itself proof of a sensor fault.",
        "- Fault-type performance varies substantially, especially for drift, stuck-at, noise, and intermittent dropout.",
        "- Open-set scores are controlled synthetic-fault evidence, not general unknown-fault detection.",
        "- Calibration metrics and uncertainty records are reported as evidence, not proof of calibrated operational probabilities.",
        "- These are research results on public telemetry and synthetic faults, not production, flight-certified, or safety-certified capability.",
        "",
        "## Reproducibility",
        "Use `scripts/run_step10_sensor_health_fdi.py` with the existing Step 10.4 final artifact directory. The manifest records source paths, seed, detector, fault types, and frozen-holdout status.",
        "",
        "## Conclusion",
        "Step 10 establishes a reproducible sensor-health/FDI evidence framework and documents the central limitation: strong in-domain or ranking metrics do not translate into reliable cross-dataset detection precision. The limitation is a result, not a defect to conceal.",
    ])
    (output / "FINAL_STEP10_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_step10(output_dir: str | Path = "reports/experiments/step10_sensor_health_fdi") -> dict[str, Any]:
    """Package verified Step 10 evidence into the isolated final milestone directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detection = _read_csv(FINAL_SOURCE, "cross_dataset_results.csv")
    isolation = _read_csv(FINAL_SOURCE, "isolation_results.csv")
    classification = _classification_summary(FINAL_SOURCE)
    open_set = _read_csv(FINAL_SOURCE, "open_set_results.csv")
    calibration = _read_csv(FINAL_SOURCE, "calibration_results.csv")
    for name in ("cross_dataset_results.csv", "isolation_results.csv", "fault_classification_results.csv",
                 "open_set_results.csv", "calibration_results.csv", "uncertainty_results.csv",
                 ):
        _copy_evidence(FINAL_SOURCE, name, output)
    leakage_audit = json.loads((FINAL_SOURCE / "leakage_audit.json").read_text(encoding="utf-8"))
    leakage_audit.update({
        "final_holdout_frozen": True,
        "source_experiment": str(FINAL_SOURCE),
        "integration_retrained": False,
        "historical_results_overwritten": False,
    })
    (output / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2), encoding="utf-8")
    _copy_evidence(HISTORICAL_SOURCE, "NORMAL_DEGRADATION.csv", output)
    classification.to_csv(output / "fault_classification_summary.csv", index=False)
    detection.to_csv(output / "final_detection_results.csv", index=False)
    isolation.groupby("dataset")[["top1", "top3", "top5", "top10", "mrr"]].mean().reset_index().to_csv(
        output / "final_isolation_summary.csv", index=False
    )
    _write_plots(detection, isolation, output)
    manifest = {
        "step": "10",
        "title": "Sensor Health / FDI",
        "integration_type": "evidence packaging of verified historical Step 10.4 final run; no retraining",
        "source_experiment": str(FINAL_SOURCE),
        "historical_normal_degradation_source": str(HISTORICAL_SOURCE / "NORMAL_DEGRADATION.csv"),
        "datasets": ["FD001", "FD002", "FD003", "FD004"],
        "seed": 42,
        "detector": "condition residual + soft temporal evidence + peer consistency",
        "fault_types": ["BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT", "SPIKE", "SATURATION", "GAIN_ERROR", "MISSING_BLOCK", "BURST_NOISE"],
        "final_holdout_frozen": True,
        "historical_artifacts_preserved": True,
        "source_files": [
            str(FINAL_SOURCE / name) for name in (
                "cross_dataset_results.csv", "isolation_results.csv",
                "fault_classification_results.csv", "open_set_results.csv",
                "calibration_results.csv", "uncertainty_results.csv", "leakage_audit.json",
            )
        ],
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(detection, isolation, classification, open_set, calibration, output)
    return manifest
