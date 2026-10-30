"""Run the AERIS-X Step 7 classical ML benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.evaluation.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/cmapss_fd001")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports/experiments/step7_classical_ml"
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    results = run_benchmark(args.data_dir, args.output_dir, args.random_state)
    print(results.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
