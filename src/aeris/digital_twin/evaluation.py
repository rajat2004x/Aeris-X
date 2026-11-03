"""Lightweight evaluation of synthetic fault detection and isolation."""

from __future__ import annotations

from typing import Any

import numpy as np

from aeris.digital_twin.generator import SENSOR_NAMES, generate_scenario


FAULTS = ("BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT",
          "COMPRESSOR_DEGRADATION", "TURBINE_DEGRADATION", "EFFICIENCY_LOSS",
          "TEMPERATURE_DEGRADATION", "PRESSURE_DEGRADATION")


def evaluate_synthetic(seed: int = 42) -> dict[str, Any]:
    rows = []
    for index, fault in enumerate(("NORMAL",) + FAULTS):
        telemetry, metadata = generate_scenario(fault, seed=seed + index)
        baseline = telemetry[:60]
        center = np.nanmedian(baseline, axis=0)
        scale = np.nanstd(baseline, axis=0) + 1e-6
        point_scores = np.nanmax(np.abs((telemetry - center) / scale), axis=1)
        channel_scores = np.nanmax(np.abs((telemetry - center) / scale), axis=0)
        detected = bool(np.nanmax(point_scores[60:]) > 5.0)
        ranking = np.argsort(channel_scores)[::-1]
        target = metadata["affected_sensor"]
        target_index = SENSOR_NAMES.index(target) if target else None
        rows.append({
            "scenario_id": metadata["scenario_id"],
            "fault_type": fault,
            "detected": detected,
            "target_sensor": target,
            "top1": bool(target_index is not None and ranking[0] == target_index),
            "top3": bool(target_index is not None and target_index in ranking[:3]),
            "top5": bool(target_index is not None and target_index in ranking[:5]),
            "ground_truth_fault": bool(metadata["ground_truth"]["fault"]),
            "ground_truth_degradation": bool(metadata["ground_truth"]["degradation"]),
        })
    return {"rows": rows, "faults": list(FAULTS), "scope": "controlled synthetic telemetry for research and robustness evaluation"}
