"""Deterministic Phase 16 scenario matrix with split and leakage metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FAULT_TYPES = ("BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT")


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    split: str
    seed: int
    fault_type: str
    affected_sensor: int | None
    magnitude: float
    start_time: int
    duration: int
    operating_condition: dict[str, float]


def build_scenario_matrix(
    seed: int = 42,
    sensors: int = 8,
    start_time: int = 60,
    duration: int = 60,
) -> list[ScenarioSpec]:
    if sensors < 1 or start_time < 1 or duration < 1:
        raise ValueError("sensors, start_time, and duration must be positive")
    rows: list[ScenarioSpec] = [
        ScenarioSpec(
            scenario_id="normal_holdout_000",
            split="heldout_evaluation",
            seed=seed,
            fault_type="NORMAL",
            affected_sensor=None,
            magnitude=0.0,
            start_time=start_time,
            duration=0,
            operating_condition={"throttle": 0.55},
        )
    ]
    for index, fault_type in enumerate(FAULT_TYPES):
        rows.append(ScenarioSpec(
            scenario_id=f"heldout_{fault_type.lower()}_{index:03d}",
            split="heldout_evaluation",
            seed=seed + index + 1,
            fault_type=fault_type,
            affected_sensor=index % sensors,
            magnitude=1.0,
            start_time=start_time,
            duration=duration,
            operating_condition={"throttle": 0.55 + 0.05 * (index % 2)},
        ))
    return rows


def scenario_manifest(seed: int = 42) -> dict[str, Any]:
    scenarios = build_scenario_matrix(seed=seed)
    return {
        "experiment_version": "phase16-scenarios-v1",
        "data_type": "synthetic_digital_twin_only",
        "seed": seed,
        "scenarios": [asdict(item) for item in scenarios],
        "training_development_scenarios": [],
        "heldout_evaluation_scenarios": [item.scenario_id for item in scenarios],
        "threshold_tuning": "none; detector uses a documented fixed heuristic threshold",
        "synthetic_parameters_used_for_training": False,
    }
