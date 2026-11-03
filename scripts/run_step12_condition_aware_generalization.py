"""Run the isolated Step 12 condition-aware generalization study."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris.health.condition_aware import run_condition_aware


if __name__ == "__main__":
    result = run_condition_aware()
    print(result["transfer"][["source", "target", "mae", "rmse", "r2", "spearman_correlation"]].to_string(index=False))
