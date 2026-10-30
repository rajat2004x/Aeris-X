"""Enforce that a requested milestone is the next authorized milestone."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_roadmap import read_milestones, validate


def check(requested: str) -> tuple[bool, str]:
    valid, message = validate()
    if not valid:
        return False, f"BLOCKED\nReason: roadmap state is invalid: {message}"
    milestones = {item["id"]: item["name"] for item in read_milestones()}
    if requested not in milestones:
        return False, f"BLOCKED\nReason: unknown milestone {requested}"
    current = next(item["id"] for item in read_milestones() if item["id"] not in __import__("json").loads((ROOT / ".aeris" / "state.json").read_text())["completed_steps"])
    if requested != current:
        return False, f"BLOCKED\nReason: Step {current} must be completed before Step {requested}."
    return True, f"ALLOWED\nStep {requested} - {milestones[requested]}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_build_gate.py <milestone>", file=sys.stderr)
        sys.exit(2)
    allowed, message = check(sys.argv[1])
    print(message)
    sys.exit(0 if allowed else 1)
