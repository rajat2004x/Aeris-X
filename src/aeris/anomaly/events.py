"""Temporal persistence and event segmentation."""

from __future__ import annotations

import numpy as np


def persistence_flags(flags: np.ndarray, consecutive: int) -> np.ndarray:
    """Require consecutive positive points before raising an alarm."""
    values = np.asarray(flags, dtype=bool)
    if values.ndim != 1 or consecutive < 1:
        raise ValueError("flags must be 1-D and consecutive must be positive.")
    output = np.zeros_like(values)
    run = 0
    for index, flag in enumerate(values):
        run = run + 1 if flag else 0
        if run >= consecutive:
            output[index] = True
    return output


def event_segments(flags: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end indices for contiguous positive events."""
    values = np.asarray(flags, dtype=bool)
    if values.ndim != 1:
        raise ValueError("flags must be 1-D.")
    padded = np.concatenate(([False], values, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def event_metrics(labels: np.ndarray, alarms: np.ndarray) -> dict[str, float | int]:
    """Measure event recall, false alarms, and mean detection delay."""
    actual = event_segments(labels)
    predicted = event_segments(alarms)
    matched = []
    delays = []
    for start, end in actual:
        hits = [event for event in predicted if event[0] <= end and event[1] >= start]
        if hits:
            first = min(hits, key=lambda event: event[0])
            matched.append(first)
            delays.append(max(0, first[0] - start))
    false_alarms = max(0, len(predicted) - len(matched))
    return {
        "actual_events": len(actual),
        "detected_events": len(matched),
        "event_recall": len(matched) / max(1, len(actual)),
        "false_alarms": false_alarms,
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
    }
