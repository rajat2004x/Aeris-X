"""Report the one milestone authorized for the next autonomous build."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
state = json.loads((ROOT / ".aeris" / "state.json").read_text(encoding="utf-8"))
print(f"NEXT AUTHORIZED MILESTONE: {state['current_step']} - {state['current_step_name']}")
print("This control layer does not implement the milestone.")
