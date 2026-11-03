"""Bounded telemetry quality and reference-profile comparisons."""

from typing import Sequence

import numpy as np

from aeris.domain.schemas import DataQualityResult, DriftComparison, ReferenceProfile


def evaluate_window(window: Sequence[Sequence[float]], feature_names: list[str] | None = None) -> DataQualityResult:
    values = np.asarray(window, dtype=float)
    if values.ndim != 2 or values.size == 0:
        return DataQualityResult(valid=False, range_violations=["window must be a non-empty matrix"])
    names = feature_names or [f"feature_{i}" for i in range(values.shape[1])]
    statistics = {}
    for index, name in enumerate(names):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        statistics[name] = {
            "count": int(column.size),
            "missing_count": int(column.size - finite.size),
            "mean": float(finite.mean()) if finite.size else 0.0,
            "std": float(finite.std()) if finite.size else 0.0,
            "min": float(finite.min()) if finite.size else 0.0,
            "max": float(finite.max()) if finite.size else 0.0,
        }
    return DataQualityResult(
        valid=bool(np.isfinite(values).all()),
        non_finite_values=[] if np.isfinite(values).all() else ["window"],
        sequence_length=values.shape[0],
        channel_count=values.shape[1],
        feature_statistics=statistics,
    )


def compare_to_reference(
    window: Sequence[Sequence[float]], profile: ReferenceProfile
) -> list[DriftComparison]:
    values = np.asarray(window, dtype=float)
    results = []
    for index, feature in enumerate(profile.feature_names):
        if index >= values.shape[1] or feature not in profile.means:
            results.append(DriftComparison(
                feature=feature, comparable=False, status="NOT_COMPARABLE",
                note="reference statistics unavailable"
            ))
            continue
        observed = values[:, index]
        observed_mean = float(np.mean(observed))
        reference = profile.means[feature]
        results.append(DriftComparison(
            feature=feature,
            reference_mean=reference,
            observed_mean=observed_mean,
            mean_delta=observed_mean - reference,
            comparable=True,
            status="COMPARED",
            note="descriptive comparison only; no alert threshold applied",
        ))
    return results
