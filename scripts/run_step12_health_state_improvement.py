"""Run controlled Step 12 health-estimation improvements without changing the baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aeris.data.cmapss import load_cmapss, make_windows
from aeris.health.improvement import (
    _fit_candidate,
    evaluate_predictions,
    predict_transformer,
    robustness_evaluation,
)
from aeris.health.labels import rul_to_health
from aeris.health.trajectory import split_by_engine_order


def _ownership(root: Path, metadata: dict[str, object], engine_ids: list[int]) -> np.ndarray:
    dataset = load_cmapss(root / "data" / "raw", "FD001")
    _, owners = make_windows(dataset.train, engine_ids, metadata["feature_columns"], metadata["window_size"])
    return owners


def run_improvement(
    data_dir: str | Path = "data/processed/cmapss_fd001",
    output_dir: str | Path = "reports/experiments/step12_health_state_improvement",
    seed: int = 42,
    epochs: int = 3,
) -> dict[str, object]:
    """Run baseline comparison and two predeclared candidate experiments."""
    root = PROJECT_ROOT
    data = root / data_dir
    output = root / output_dir
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    X_train = np.load(data / "X_train.npy")
    y_train_rul = np.load(data / "y_train.npy").astype(float)
    X_validation = np.load(data / "X_validation.npy")
    y_validation_rul = np.load(data / "y_validation.npy").astype(float)
    max_rul = float(y_train_rul.max())
    y_train = rul_to_health(y_train_rul, max_rul)
    y_validation = rul_to_health(y_validation_rul, max_rul)
    train_owners = _ownership(root, metadata, metadata["train_engines"])
    validation_owners = _ownership(root, metadata, metadata["validation_engines"])
    dev_mask, final_mask, dev_engines, final_engines = split_by_engine_order(validation_owners)
    baseline_checkpoint = root / "reports" / "experiments" / "step8_temporal_dl" / "models" / "transformer.pt"
    baseline_prediction = rul_to_health(
        predict_transformer(X_validation, baseline_checkpoint), max_rul
    )
    baseline_results = pd.DataFrame([
        evaluate_predictions(y_validation[dev_mask], baseline_prediction[dev_mask], validation_owners[dev_mask], "development", "baseline_transformer"),
        evaluate_predictions(y_validation[final_mask], baseline_prediction[final_mask], validation_owners[final_mask], "final", "baseline_transformer"),
    ])
    baseline_results.to_csv(output / "baseline_results.csv", index=False)
    models_dir = output / "models"
    candidate_predictions: dict[str, np.ndarray] = {}
    candidate_rows: list[dict[str, object]] = []
    for candidate in ("lifecycle_huber", "temporal_consistency"):
        checkpoint = models_dir / f"{candidate}.pt"
        training = _fit_candidate(X_train, y_train, train_owners, candidate, checkpoint, seed, epochs=epochs)
        prediction = predict_transformer(X_validation, checkpoint)
        candidate_predictions[candidate] = prediction
        dev_result = evaluate_predictions(y_validation[dev_mask], prediction[dev_mask], validation_owners[dev_mask], "development", candidate)
        dev_result.update(training)
        candidate_rows.append(dev_result)
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(output / "candidate_results.csv", index=False)
    selection = candidates.sort_values(["mae", "rmse", "model"]).iloc[0]
    selected = str(selection["model"])
    selected_prediction = candidate_predictions[selected]
    final_rows = [
        evaluate_predictions(y_validation[final_mask], baseline_prediction[final_mask], validation_owners[final_mask], "final", "baseline_transformer"),
        evaluate_predictions(y_validation[final_mask], selected_prediction[final_mask], validation_owners[final_mask], "final", selected),
    ]
    final_results = pd.DataFrame(final_rows)
    final_results.to_csv(output / "final_results.csv", index=False)
    trajectory_rows = []
    for model_name, prediction in (("baseline_transformer", baseline_prediction), (selected, selected_prediction)):
        for engine_id in final_engines:
            selected_engine = final_mask & (validation_owners == engine_id)
            trajectory_rows.append({
                "model": model_name,
                "engine_id": engine_id,
                "windows": int(selected_engine.sum()),
                **evaluate_predictions(
                    y_validation[selected_engine], prediction[selected_engine],
                    validation_owners[selected_engine], "final", model_name,
                ),
            })
    pd.DataFrame(trajectory_rows).to_csv(output / "trajectory_results.csv", index=False)
    training_std = np.std(X_train.reshape(-1, X_train.shape[-1]), axis=0)
    robustness = robustness_evaluation(
        X_validation[final_mask], y_validation[final_mask], validation_owners[final_mask],
        baseline_prediction[final_mask], selected_prediction[final_mask],
        baseline_checkpoint, models_dir / f"{selected}.pt", training_std,
        max_rul,
    )
    pd.DataFrame(robustness).to_csv(output / "robustness_results.csv", index=False)

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plot_data = final_results.set_index("model")
    axis = plot_data[["mae", "rmse"]].plot(kind="bar", title="Baseline versus selected candidate")
    axis.set_ylabel("error")
    axis.figure.tight_layout()
    axis.figure.savefig(plots / "baseline_vs_candidate.png", dpi=150)
    plt.close(axis.figure)
    selected_engine = final_mask & (validation_owners == final_engines[0])
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(y_validation[selected_engine], label="true health")
    axis.plot(baseline_prediction[selected_engine], label="baseline")
    axis.plot(selected_prediction[selected_engine], label=selected)
    axis.legend()
    axis.set(title=f"Health trajectory comparison, engine {final_engines[0]}", xlabel="window", ylabel="health")
    figure.tight_layout()
    figure.savefig(plots / "health_trajectory_comparison.png", dpi=150)
    plt.close(figure)
    axis = plot_data["monotonicity_violations"].plot(kind="bar", title="Monotonicity violations")
    axis.set_ylabel("count")
    axis.figure.tight_layout()
    axis.figure.savefig(plots / "monotonicity_comparison.png", dpi=150)
    plt.close(axis.figure)
    lifecycle = plot_data[["early_life_mae", "mid_life_mae", "late_life_mae"]].plot(kind="bar", title="Lifecycle error comparison")
    lifecycle.set_ylabel("MAE")
    lifecycle.figure.tight_layout()
    lifecycle.figure.savefig(plots / "lifecycle_error_comparison.png", dpi=150)
    plt.close(lifecycle.figure)
    robustness_frame = pd.DataFrame(robustness).set_index("model")
    axis = robustness_frame[["mae_degradation", "prediction_stability"]].plot(kind="bar", title="Robustness under sensor corruption")
    axis.set_ylabel("change")
    axis.figure.tight_layout()
    axis.figure.savefig(plots / "robustness_comparison.png", dpi=150)
    plt.close(axis.figure)

    baseline_final = final_results[final_results["model"] == "baseline_transformer"].iloc[0]
    candidate_final = final_results[final_results["model"] == selected].iloc[0]
    accepted = bool(
        candidate_final["mae"] < baseline_final["mae"]
        and candidate_final["rmse"] <= baseline_final["rmse"] * 1.02
        and candidate_final["r2"] >= baseline_final["r2"] - 0.02
        and candidate_final["spearman_correlation"] >= baseline_final["spearman_correlation"] - 0.02
        and candidate_final["monotonicity_violations"] <= baseline_final["monotonicity_violations"]
    )
    manifest = {
        "step": "12 improvement",
        "dataset": "NASA C-MAPSS FD001",
        "baseline_artifacts_immutable": True,
        "health_definition": f"clip(RUL / {max_rul}, 0, 1)",
        "development_engines": dev_engines,
        "final_engines": final_engines,
        "candidates": ["lifecycle_huber", "temporal_consistency"],
        "selection_rule": "lowest development MAE, then RMSE, then model name",
        "acceptance_rule": "lower MAE, no material RMSE/R2/Spearman regression, and no increase in monotonicity violations",
        "selected_candidate": selected,
        "accepted": accepted,
        "uncertainty": "Not operationally calibrated; no new uncertainty method tested.",
        "robustness": "evaluation-only sensor_0 BIAS injection on final holdout windows",
    }
    (output / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "leakage_audit.json").write_text(json.dumps({
        "training": "X_train and y_train only",
        "development": "development validation engines only for candidate comparison",
        "final": "final validation engines evaluated once after selection",
        "final_holdout_used_for_training": False,
        "final_holdout_used_for_selection": False,
        "target_reference_max_fit_on_training_only": True,
        "engine_boundaries_preserved": True,
        "robustness_injection_used_for_training": False,
        "baseline_overwritten": False,
    }, indent=2), encoding="utf-8")
    decision = "Step 12 materially improved." if accepted else "No candidate materially improved the existing Step 12 baseline; baseline retained."
    (output / "FINAL_STEP12_IMPROVEMENT_REPORT.md").write_text(
        f"""# Step 12 Improvement Experiment

## Existing baseline
The immutable control is the existing Step 12 Transformer. Its final metrics are
preserved in `baseline_results.csv` and the original Step 12 directory.

## Weaknesses and hypotheses
The baseline had 479 monotonicity violations, 0.568306 direction consistency, and
early-life MAE 0.104508 versus late-life MAE 0.010781. Candidate A tests lifecycle
balanced Huber loss. Candidate B tests a 0.10 temporal consistency penalty only for
same-engine adjacent windows whose true health is non-increasing. Neither forces all
local predictions to decrease.

## Experiments and selection
Only the two predeclared candidates were trained. Selection used development engines
{dev_engines} and the stated MAE/RMSE/model-name rule. Final engines {final_engines}
were frozen until after selection.

## Results

{final_results.to_string(index=False)}

## Monotonicity and lifecycle analysis
Candidate metrics include violations, direction consistency, smoothness, and
early/mid/late-life MAE. A lower error candidate is not accepted if trajectory
consistency materially worsens.

## Robustness

{pd.DataFrame(robustness).to_string(index=False)}

The corruption experiment is evaluation-only and does not establish FDI or anomaly
 detection capability. The frozen final subset contains no windows at the fixed
 healthy threshold (0.75), so healthy-as-degraded rate is not applicable there.

## Uncertainty
Not operationally calibrated. No new uncertainty claim is made.

## Leakage and reproducibility
See `leakage_audit.json`, `config.json`, and `experiment_manifest.json`. Training,
development, and frozen-final engine boundaries are preserved. Seed: {seed}.

## Final decision
**{decision}**

Baseline Step 12 artifacts were not overwritten. Step 13 was not implemented and the
roadmap was not advanced.
""",
        encoding="utf-8",
    )
    return {"selected": selected, "accepted": accepted, "final_results": final_results, "robustness": robustness}


if __name__ == "__main__":
    result = run_improvement()
    print({"selected": result["selected"], "accepted": result["accepted"]})
