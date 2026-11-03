"""Generate bounded Phase 16 intelligence artifacts without model training."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from aeris.digital_twin.generator import generate_normal, inject_scenario
from aeris.intelligence.detectors import RankedDeviationIsolator, RobustWindowAnomalyDetector
from aeris.intelligence.orchestration import IntelligenceOrchestrator
from aeris.intelligence.scenarios import build_scenario_matrix, scenario_manifest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output_root: str | Path = "reports/experiments") -> dict[str, object]:
    root = Path(output_root)
    intelligence = root / "phase16_intelligence"
    fault_injection = root / "phase16_fault_injection"
    diagnosis = root / "phase16_diagnosis"
    for directory in (intelligence, fault_injection, diagnosis):
        directory.mkdir(parents=True, exist_ok=True)

    seed = 42
    manifest = scenario_manifest(seed)
    (fault_injection / "scenario_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    audit = {
        "experiment_version": "phase16-foundation-v1",
        "dataset": "synthetic digital-twin telemetry",
        "real_nasa_data_used": False,
        "training_split": "none; no model training",
        "development_split": "none; no threshold calibration",
        "final_evaluation_split": "heldout_evaluation scenarios only",
        "final_labels_used_for_selection": False,
        "synthetic_fault_parameters_used_for_training": False,
        "scaler_fit_scope": "none",
        "random_seed": seed,
        "threshold": 3.0,
        "threshold_source": "fixed documented heuristic; not tuned on evaluation labels",
    }
    for directory in (intelligence, fault_injection, diagnosis):
        (directory / "leakage_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )

    detector = RobustWindowAnomalyDetector()
    isolator = RankedDeviationIsolator()
    intelligence_orchestrator = IntelligenceOrchestrator()
    rows: list[dict[str, object]] = []
    for spec in build_scenario_matrix(seed=seed):
        telemetry, _ = generate_normal(length=120, seed=spec.seed)
        if spec.fault_type != "NORMAL":
            telemetry, scenario = inject_scenario(
                telemetry,
                spec.fault_type,
                sensor=int(spec.affected_sensor),
                start_time=spec.start_time,
                severity=spec.magnitude,
                seed=spec.seed,
            )
            # Dropout is expected missing telemetry, not a detector crash.
            telemetry = np.nan_to_num(telemetry, nan=0.0)
            affected = f"sensor_{spec.affected_sensor}"
        else:
            affected = None
        names = [f"sensor_{index}" for index in range(telemetry.shape[1])]
        anomaly = detector.detect(telemetry, names)
        isolation = isolator.isolate(telemetry, names)
        combined = intelligence_orchestrator.analyze(
            asset_id=spec.scenario_id,
            window=telemetry,
            feature_names=names,
            operating_condition=spec.operating_condition,
        )
        top1 = bool(affected is not None and isolation.ranked_sensors[0] == affected)
        rows.append({
            "scenario_id": spec.scenario_id,
            "split": spec.split,
            "dataset": "synthetic_digital_twin",
            "fault_type": spec.fault_type,
            "affected_sensor": affected,
            "anomaly_status": anomaly.status,
            "anomaly_score": anomaly.score,
            "threshold": anomaly.threshold,
            "top1_isolation": top1,
            "detector": anomaly.detector,
            "diagnosis_state": combined.diagnosis.state.value,
            "health_score": combined.health_state.health_score,
            "seed": spec.seed,
        })
    _write_csv(fault_injection / "scenario_results.csv", rows)
    _write_csv(intelligence / "combined_intelligence_results.csv", rows)
    _write_csv(diagnosis / "diagnosis_results.csv", rows)
    positives = [row for row in rows if row["fault_type"] != "NORMAL"]
    predicted = [row["anomaly_status"] == "ANOMALOUS" for row in rows]
    actual = [row["fault_type"] != "NORMAL" for row in rows]
    true_positive = sum(p and a for p, a in zip(predicted, actual))
    false_positive = sum(p and not a for p, a in zip(predicted, actual))
    false_negative = sum(not p and a for p, a in zip(predicted, actual))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    top1 = sum(bool(row["top1_isolation"]) for row in positives) / max(1, len(positives))
    report = f"""# Phase 16 Foundation Report

## Sub-phase status

| Sub-phase | Status | Evidence |
|---|---|---|
| 16A — Intelligence domain contracts | COMPLETE | `src/aeris/intelligence/contracts.py` |
| 16B — Unified aerospace health state | COMPLETE | `src/aeris/intelligence/orchestration.py` |
| 16C — Advanced anomaly/FDI orchestration | COMPLETE | detector and isolator protocols |
| 16D — Subsystem diagnosis | COMPLETE | deterministic rule set `phase16-deterministic-v1` |
| 16E — Digital-twin fault scenarios | COMPLETE | `scenario_manifest.json` |
| 16F — Evidence foundation | COMPLETE | structured evidence items, no LLM |
| 16G — Evaluation and integration | COMPLETE | API, scenario results, leakage audit |

## Bounded synthetic evaluation

The following values are measured on the six held-out synthetic digital-twin
scenarios generated by this run. They are not real NASA validation:

- anomaly precision: {precision:.4f}
- anomaly recall: {recall:.4f}
- anomaly F1: {f1:.4f}
- sensor isolation Top-1: {top1:.4f}
- scenarios: {len(rows)}

This bounded package composes existing AERIS-X research interfaces and evaluates
deterministic rules on synthetic digital-twin fault scenarios. It does not claim
real NASA validation, calibrated confidence, general fault diagnosis, or flight
readiness. No model was retrained and no historical experiment directory was
modified.

The Transformer health path remains the primary production health reference when
the API is used. The scenario matrix is held out and uses a fixed heuristic
threshold; it is not used for model selection or threshold tuning.
"""
    (intelligence / "FINAL_PHASE16_REPORT.md").write_text(report, encoding="utf-8")
    return {"scenario_count": len(rows), "output_directories": [str(intelligence), str(fault_injection), str(diagnosis)]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
