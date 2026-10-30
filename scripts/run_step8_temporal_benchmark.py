"""Run the AERIS-X Step 8 temporal deep-learning benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.evaluation.temporal_benchmark import run_temporal_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/cmapss_fd001")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/experiments/step8_temporal_dl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    results = run_temporal_benchmark(args.data_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size)
    print(results.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
