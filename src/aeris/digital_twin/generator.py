"""Deterministic synthetic telemetry for research and robustness evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


SENSOR_NAMES = (
    "compressor_pressure",
    "turbine_temperature",
    "shaft_speed",
    "fuel_flow",
    "oil_pressure",
    "vibration",
    "ambient_temperature",
    "ambient_pressure",
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    fault_type: str
    affected_sensor: str | None
    affected_subsystem: str | None
    start_time: int
    duration: int
    severity: float
    operating_condition: dict[str, float]
    ground_truth: dict[str, Any]


def generate_normal(length: int = 120, seed: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    if length < 10:
        raise ValueError("length must be at least 10")
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    condition = {
        "throttle": float(rng.uniform(0.35, 0.8)),
        "ambient_temperature": float(rng.normal(288.0, 3.0)),
        "ambient_pressure": float(rng.normal(101.3, 1.0)),
    }
    base = np.column_stack([
        1.0 + 0.04 * np.sin(t / 10),
        0.8 + 0.03 * np.cos(t / 12),
        0.6 + 0.02 * np.sin(t / 8),
        0.5 + 0.02 * np.cos(t / 9),
        0.7 + 0.015 * np.sin(t / 7),
        0.2 + 0.01 * np.cos(t / 6),
        np.full(length, condition["ambient_temperature"] / 300),
        np.full(length, condition["ambient_pressure"] / 100),
    ])
    return (base + rng.normal(0, 0.005, base.shape)).astype(np.float32), condition


def inject_scenario(
    telemetry: np.ndarray,
    fault_type: str,
    sensor: int = 0,
    start_time: int = 60,
    severity: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, Scenario]:
    values = np.asarray(telemetry, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(SENSOR_NAMES):
        raise ValueError("telemetry must have shape [time, 8].")
    if not 0 <= sensor < values.shape[1] or not 0 <= start_time < len(values):
        raise ValueError("sensor and start_time must be in range.")
    if severity <= 0:
        raise ValueError("severity must be positive.")
    result = values.copy()
    tail = result[start_time:, sensor]
    if fault_type == "BIAS":
        tail += 0.1 * severity
        subsystem = "sensor"
    elif fault_type == "DRIFT":
        tail += np.linspace(0, 0.2 * severity, len(tail))
        subsystem = "sensor"
    elif fault_type == "STUCK_AT":
        tail[:] = tail[0]
        subsystem = "sensor"
    elif fault_type == "NOISE":
        tail += np.random.default_rng(seed).normal(0, 0.05 * severity, len(tail))
        subsystem = "sensor"
    elif fault_type == "INTERMITTENT_DROPOUT":
        mask = np.random.default_rng(seed).random(len(tail)) < min(0.8, 0.2 * severity)
        tail[mask] = np.nan
        subsystem = "sensor"
    elif fault_type == "COMPRESSOR_DEGRADATION":
        result[start_time:, 0] -= 0.15 * severity
        subsystem = "compressor"
    elif fault_type == "TURBINE_DEGRADATION":
        result[start_time:, 1] += 0.15 * severity
        subsystem = "turbine"
    elif fault_type == "EFFICIENCY_LOSS":
        result[start_time:, 0:4] *= 1.0 - 0.1 * severity
        subsystem = "engine"
    elif fault_type == "TEMPERATURE_DEGRADATION":
        result[start_time:, 1] += 0.2 * severity
        subsystem = "thermal"
    elif fault_type == "PRESSURE_DEGRADATION":
        result[start_time:, [0, 4]] -= 0.15 * severity
        subsystem = "pressure"
    else:
        raise ValueError(f"Unsupported scenario: {fault_type}")
    scenario = Scenario(
        scenario_id=f"{fault_type.lower()}_{seed}_{start_time}",
        seed=seed,
        fault_type=fault_type,
        affected_sensor=SENSOR_NAMES[sensor] if fault_type in {"BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"} else None,
        affected_subsystem=subsystem,
        start_time=start_time,
        duration=len(values) - start_time,
        severity=severity,
        operating_condition={},
        ground_truth={"fault": fault_type != "NORMAL", "degradation": "DEGRADATION" in fault_type or fault_type in {"EFFICIENCY_LOSS"}},
    )
    return result, scenario


def generate_scenario(fault_type: str, seed: int = 42, length: int = 120) -> tuple[np.ndarray, dict[str, Any]]:
    telemetry, condition = generate_normal(length, seed)
    if fault_type == "NORMAL":
        scenario = Scenario(
            scenario_id=f"normal_{seed}", seed=seed, fault_type="NORMAL",
            affected_sensor=None, affected_subsystem=None, start_time=length,
            duration=0, severity=0.0, operating_condition=condition,
            ground_truth={"fault": False, "degradation": False},
        )
        return telemetry, asdict(scenario)
    result, scenario = inject_scenario(telemetry, fault_type, seed=seed)
    return result, asdict(scenario) | {"operating_condition": condition}
