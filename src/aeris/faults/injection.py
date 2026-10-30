"""Deterministic synthetic telemetry fault injection for controlled experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FaultInjection:
    fault_type: str
    sensor_index: int
    start_cycle: int
    severity: str
    seed: int


def inject_fault(
    sequence: np.ndarray,
    sensor_index: int,
    fault_type: str,
    onset: float,
    severity: float,
    training_std: float,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a copied sequence with one controlled synthetic fault."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or not 0 <= sensor_index < values.shape[1]:
        raise ValueError("sequence must be 2-D and sensor_index must be valid.")
    if not 0 <= onset < 1:
        raise ValueError("onset must be in [0, 1).")
    if training_std <= 0 or severity <= 0:
        raise ValueError("training_std and severity must be positive.")
    result = values.copy()
    start = min(values.shape[0] - 1, max(0, int(values.shape[0] * onset)))
    scale = float(training_std) * float(severity)
    rng = np.random.default_rng(seed)
    tail = result[start:, sensor_index]
    if fault_type == "BIAS":
        result[start:, sensor_index] += scale
    elif fault_type == "DRIFT":
        result[start:, sensor_index] += np.linspace(0, scale, len(tail), dtype=np.float32)
    elif fault_type == "STUCK_AT":
        result[start:, sensor_index] = result[start, sensor_index]
    elif fault_type == "NOISE":
        result[start:, sensor_index] += rng.normal(0, scale, size=len(tail)).astype(np.float32)
    elif fault_type == "INTERMITTENT_DROPOUT":
        mask = rng.random(len(tail)) < min(0.95, severity)
        result[start:, sensor_index][mask] = result[start, sensor_index]
    elif fault_type == "SPIKE":
        count = max(1, int(len(tail) * min(0.5, severity)))
        indices = rng.choice(len(tail), count, replace=False)
        result[start:, sensor_index][indices] += scale * 3
    elif fault_type == "SATURATION":
        limit = result[start, sensor_index] + scale
        if np.max(result[start:, sensor_index]) <= limit:
            limit = result[start, sensor_index] - scale
            result[start:, sensor_index] = np.minimum(result[start:, sensor_index], limit)
        else:
            result[start:, sensor_index] = np.minimum(result[start:, sensor_index], limit)
    elif fault_type == "GAIN_ERROR":
        result[start:, sensor_index] *= 1.0 + severity
    elif fault_type == "OFFSET_ERROR":
        result[start:, sensor_index] -= scale
    elif fault_type == "DEADBAND":
        delta = result[start:, sensor_index] - result[start, sensor_index]
        result[start:, sensor_index] = np.where(np.abs(delta) < scale, result[start, sensor_index], result[start:, sensor_index])
    elif fault_type == "QUANTIZATION":
        step = max(scale / 2, 1e-6)
        result[start:, sensor_index] = np.round(result[start:, sensor_index] / step) * step
    elif fault_type == "MISSING_BLOCK":
        block = max(1, int(len(tail) * min(0.5, severity)))
        block_start = max(0, (len(tail) - block) // 2)
        result[start + block_start : start + block_start + block, sensor_index] = np.nan
    elif fault_type == "BURST_NOISE":
        block = max(1, int(len(tail) * min(0.35, severity)))
        block_start = max(0, (len(tail) - block) // 2)
        result[start + block_start : start + block_start + block, sensor_index] += rng.normal(
            0, scale * 2, size=block
        ).astype(np.float32)
    elif fault_type == "INTERMITTENT_BIAS":
        mask = rng.random(len(tail)) < min(0.8, max(0.1, severity))
        result[start:, sensor_index][mask] += scale
    elif fault_type == "ABRUPT_DRIFT":
        result[start:, sensor_index] += np.arange(len(tail), dtype=np.float32) * scale / max(1, len(tail) // 3)
    elif fault_type == "SENSOR_FREEZE":
        result[start:, sensor_index] = result[start, sensor_index]
    elif fault_type == "SENSOR_DROPOUT":
        result[start:, sensor_index] = np.nan
    elif fault_type == "COMPOSITE":
        result[start:, sensor_index] += scale
        result[start:, sensor_index] += np.linspace(0, scale, len(tail), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported fault type: {fault_type}")
    metadata = asdict(FaultInjection(fault_type, sensor_index, start, str(severity), seed))
    metadata["onset_fraction"] = onset
    metadata["severity_scale"] = scale
    metadata["fault_duration"] = int(values.shape[0] - start)
    metadata["original_mean"] = float(np.mean(values[start:, sensor_index]))
    metadata["injected_mean"] = float(np.mean(result[start:, sensor_index]))
    metadata["magnitude_of_change"] = float(np.mean(np.abs(result[start:, sensor_index] - values[start:, sensor_index])))
    return result, metadata
