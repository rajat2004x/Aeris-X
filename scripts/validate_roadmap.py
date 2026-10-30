"""Validate the authoritative AERIS-X roadmap state without external packages."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / ".aeris" / "roadmap.yaml"
STATE = ROOT / ".aeris" / "state.json"


def read_milestones() -> list[dict[str, str]]:
    milestones: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in ROADMAP.read_text(encoding="utf-8").splitlines():
        match = re.match(r'\s*-\s+id:\s+"([^"]+)"\s*$', line)
        if match:
            if current:
                milestones.append(current)
            current = {"id": match.group(1)}
            continue
        match = re.match(r'\s+name:\s+(.+?)\s*$', line)
        if match and current:
            current["name"] = match.group(1)
    if current:
        milestones.append(current)
    return milestones


def validate() -> tuple[bool, str]:
    milestones = read_milestones()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    ids = [item["id"] for item in milestones]
    completed = set(state["completed_steps"])
    pending = set(state["pending_steps"])
    overlap = completed & pending
    missing = set(ids) - completed - pending
    next_step = next((item["id"] for item in milestones if item["id"] not in completed), None)
    errors = []
    if overlap:
        errors.append(f"completed and pending overlap: {sorted(overlap)}")
    if missing:
        errors.append(f"milestones missing from state: {sorted(missing)}")
    if state["current_step"] != next_step:
        errors.append(f"current_step is {state['current_step']}, expected {next_step}")
    expected_name = next(item["name"] for item in milestones if item["id"] == next_step)
    if state["current_step_name"] != expected_name:
        errors.append("current_step_name does not match roadmap")
    if errors:
        return False, "\n".join(errors)
    completed_display = ", ".join(sorted(completed, key=lambda value: (float(value.split(".")[0]), value)))
    historical = ", ".join(state["historical_out_of_order_steps"])
    output = (
        "AERIS-X ROADMAP STATUS\n=======================\n\n"
        f"Completed:\n{completed_display}\n\n"
        f"Historical out-of-order:\n{historical}\n\n"
        f"Current milestone:\n{state['current_step']} - {state['current_step_name']}\n\n"
        f"Blocked future milestones:\n{', '.join(sorted(pending - {state['current_step']}, key=lambda value: (float(value.split('.')[0]), value)))}\n\n"
        "ROADMAP STATE: VALID"
    )
    return True, output


if __name__ == "__main__":
    valid, message = validate()
    print(message)
    sys.exit(0 if valid else 1)
