"""Bounded, resumable experiment runtime control."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentTimeout(TimeoutError):
    """Raised when an experiment exceeds its configured wall-clock budget."""


class RuntimeGuard:
    def __init__(self, output: str | Path, experiment_id: str, budget_seconds: float) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.budget_seconds = float(budget_seconds)
        self.started = time.time()
        self.status_path = self.output / f"{experiment_id}_status.json"
        self.write("RUNNING", {})

    def elapsed(self) -> float:
        return time.time() - self.started

    def check(self, progress: dict[str, Any] | None = None) -> None:
        if self.elapsed() > self.budget_seconds:
            self.write("TIMEOUT", progress or {})
            raise ExperimentTimeout(
                f"{self.experiment_id} exceeded {self.budget_seconds:.0f}s runtime budget."
            )
        if progress:
            self.write("RUNNING", progress)

    def write(self, status: str, payload: dict[str, Any]) -> None:
        self.status_path.write_text(json.dumps({
            "experiment_id": self.experiment_id,
            "status": status,
            "elapsed_seconds": self.elapsed(),
            **payload,
        }, indent=2), encoding="utf-8")

    def complete(self, payload: dict[str, Any]) -> None:
        self.write("COMPLETED", payload)
