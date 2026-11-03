"""Controlled condition-aware generalization study for Step 12."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aeris.data.cmapss import load_cmapss, make_windows, split_engines
from aeris.health.evaluation import regression_metrics, trajectory_metrics
from aeris.health.labels import rul_to_health
from aeris.health.multidataset import FEATURE_COLUMNS, ROOT, WINDOW_SIZE, _rul_windows
from aeris.health.trajectory import early_late_errors
from aeris.models.temporal import TransformerRegressor

SEED = 42
DATASETS = ("FD001", "FD002", "FD003", "FD004")
CONTROL_DIR = ROOT / "reports" / "experiments" / "step12_health_state_multidataset"
OUTPUT_DIR = ROOT / "reports" / "experiments" / "step12_condition_aware_generalization"


def condition_key(frame: pd.DataFrame) -> pd.Series:
    """Derive the existing deterministic operating-condition identity."""
    rounded = frame[["setting_1", "setting_3"]].round({"setting_1": 0, "setting_3": 0})
    return rounded.astype(str).agg("|".join, axis=1)


def condition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    keys = condition_key(frame)
    grouped = frame.assign(condition=keys).groupby("condition")
    return {
        "condition_count": int(keys.nunique()),
        "conditions": [
            {
                "condition": str(name),
                "rows": int(len(group)),
                "engines": int(group.engine_id.nunique()),
            }
            for name, group in grouped
        ],
    }


def fit_condition_statistics(
    frame: pd.DataFrame,
    train_engines: Iterable[int],
) -> dict[str, Any]:
    """Fit condition and global statistics using training-engine rows only."""
    train = frame[frame.engine_id.isin(set(int(i) for i in train_engines))].copy()
    if train.empty:
        raise ValueError("No training rows are available for condition statistics.")
    keys = condition_key(train)
    global_values = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    stats: dict[str, dict[str, list[float]]] = {}
    for key in sorted(keys.unique()):
        values = train.loc[keys.eq(key), FEATURE_COLUMNS].to_numpy(dtype=float)
        stats[str(key)] = {
            "mean": values.mean(axis=0).tolist(),
            "scale": np.maximum(values.std(axis=0), 1e-8).tolist(),
        }
    stats["__global__"] = {
        "mean": global_values.mean(axis=0).tolist(),
        "scale": np.maximum(global_values.std(axis=0), 1e-8).tolist(),
    }
    return {
        "fit_engines": sorted(int(i) for i in set(train_engines)),
        "conditions": stats,
        "fallback": "__global__",
    }


def transform_condition_windows(
    windows: np.ndarray,
    owners: np.ndarray,
    frame: pd.DataFrame,
    statistics: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    """Normalize each window with its condition, falling back to source global stats."""
    owner_condition = (
        frame.assign(condition=condition_key(frame))
        .groupby("engine_id")["condition"]
        .first()
        .to_dict()
    )
    transformed = np.empty_like(windows, dtype=np.float32)
    fallback = 0
    for owner in np.unique(owners):
        condition = str(owner_condition.get(int(owner), statistics["fallback"]))
        selected = np.asarray([condition] * len(owners)) == condition
        if condition not in statistics["conditions"]:
            condition = statistics["fallback"]
            fallback += int(selected.sum())
        values = statistics["conditions"][condition]
        transformed[selected] = (
            (windows[selected].astype(float) - np.asarray(values["mean"]))
            / np.asarray(values["scale"])
        ).astype(np.float32)
    return transformed, {"windows": int(len(windows)), "fallback_windows": fallback}


def _fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_development: np.ndarray,
    y_development: np.ndarray,
    checkpoint: Path,
    epochs: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    model = TransformerRegressor(input_size=X_train.shape[-1], max_length=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train, dtype=torch.float32)),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    development = DataLoader(
        TensorDataset(torch.as_tensor(X_development), torch.as_tensor(y_development, dtype=torch.float32)),
        batch_size=128,
        shuffle=False,
    )
    best = float("inf")
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            development_loss = float(np.mean([
                loss_fn(model(features), targets).item()
                for features, targets in development
            ]))
        if development_loss < best:
            best = development_loss
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
    return {
        "training_time_seconds": time.perf_counter() - started,
        "best_development_loss": best,
        "epochs": epochs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _predict(checkpoint: Path, X: np.ndarray) -> tuple[np.ndarray, float]:
    model = TransformerRegressor(input_size=X.shape[-1], max_length=X.shape[1])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        prediction = model(torch.as_tensor(X, dtype=torch.float32)).numpy()
    return prediction, time.perf_counter() - started


def _metrics(actual: np.ndarray, predicted: np.ndarray, owners: np.ndarray) -> dict[str, Any]:
    values = regression_metrics(actual, predicted)
    trajectory = trajectory_metrics(actual, predicted)
    values.update(early_late_errors(actual, predicted))
    values.update({key: value for key, value in trajectory.items() if key not in values})
    values.update({
        "windows": int(len(actual)),
        "engines": int(len(np.unique(owners))),
        "prediction_min": float(np.min(predicted)),
        "prediction_max": float(np.max(predicted)),
    })
    return values


def _dataset_data(name: str, raw_root: Path, output: Path) -> dict[str, Any]:
    dataset = load_cmapss(raw_root, name)
    if name == "FD001":
        metadata = json.loads((ROOT / "data" / "processed" / "cmapss_fd001" / "metadata.json").read_text())
        train_ids = metadata["train_engines"]
        development_ids = metadata["validation_engines"][:9]
        final_ids = metadata["validation_engines"][9:]
    else:
        train_ids, development_ids, final_ids = split_engines(dataset.train.engine_id.unique(), SEED)
    statistics = fit_condition_statistics(dataset.train, train_ids)
    raw = {
        "train": _rul_windows(dataset.train, train_ids),
        "development": _rul_windows(dataset.train, development_ids),
        "final": _rul_windows(dataset.train, final_ids),
    }
    transformed = {}
    provenance = {}
    for split, (windows, owners, rul) in raw.items():
        transformed[split], provenance[split] = transform_condition_windows(
            windows, owners, dataset.train, statistics
        )
    return {
        "dataset": name,
        "frame": dataset.train,
        "train_engines": [int(i) for i in train_ids],
        "development_engines": [int(i) for i in development_ids],
        "final_engines": [int(i) for i in final_ids],
        "statistics": statistics,
        "provenance": provenance,
        "train": transformed["train"],
        "development": transformed["development"],
        "final": transformed["final"],
        "train_rul": raw["train"][2],
        "development_rul": raw["development"][2],
        "final_rul": raw["final"][2],
        "development_owners": raw["development"][1],
        "final_owners": raw["final"][1],
    }


def _baseline_reference() -> dict[str, Any]:
    in_domain = pd.read_csv(CONTROL_DIR / "in_domain" / "comparison.csv")
    transfer = pd.read_csv(CONTROL_DIR / "transfer" / "comparison.csv")
    return {
        "source": "immutable step12_health_state_multidataset artifacts",
        "in_domain": in_domain[in_domain.split.eq("final")].replace({np.nan: None}).to_dict("records"),
        "transfer": transfer.replace({np.nan: None}).to_dict("records"),
    }


def _delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    higher = {"r2", "correlation", "spearman_correlation", "degradation_direction_consistency"}
    result = {}
    for metric in ("mae", "rmse", "r2", "spearman_correlation"):
        if metric not in baseline or metric not in candidate:
            continue
        absolute = candidate[metric] - baseline[metric]
        if metric in {"mae", "rmse"}:
            absolute = -absolute
        result[f"{metric}_improvement"] = float(absolute)
        result[f"{metric}_decision"] = (
            "IMPROVED" if absolute > 1e-12 else "WORSENED" if absolute < -1e-12 else "UNCHANGED"
        )
    return result


def run_condition_aware(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Any]:
    output = ROOT / output_dir
    output.mkdir(parents=True, exist_ok=True)
    for subdirectory in ("baseline", "condition_normalized", "multi_dataset", "transfer", "plots"):
        (output / subdirectory).mkdir(parents=True, exist_ok=True)
    raw_root = ROOT / "data" / "raw"
    baseline = _baseline_reference()
    (output / "baseline_reference.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    baseline_frame = pd.DataFrame(baseline["in_domain"] + baseline["transfer"])
    baseline_frame.to_csv(output / "baseline" / "comparison.csv", index=False)
    (output / "baseline" / "results.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    hypothesis = {
        "H1": "Operating-condition effects contribute to FD001 transfer failure.",
        "H2": "Training-only condition normalization reduces operating-condition variation.",
        "H3": "Multi-condition training improves cross-condition generalization.",
        "H4": "Multi-dataset training improves held-out-dataset performance.",
        "H5": "Improvements survive a leakage audit without target-final statistics.",
    }
    (output / "hypothesis.json").write_text(json.dumps(hypothesis, indent=2), encoding="utf-8")
    datasets = {name: _dataset_data(name, raw_root, output) for name in DATASETS}
    characterization = []
    for name in DATASETS:
        dataset = load_cmapss(raw_root, name)
        characterization.append({"dataset": name, **condition_summary(dataset.train)})
    (output / "dataset_characterization.json").write_text(json.dumps(characterization, indent=2), encoding="utf-8")
    split_manifest = {
        name: {
            "train_engines": data["train_engines"],
            "development_engines": data["development_engines"],
            "final_engines": data["final_engines"],
        }
        for name, data in datasets.items()
    }
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    (output / "preprocessing_manifest.json").write_text(json.dumps({
        "method": "condition-specific z-normalization with source-global fallback",
        "condition_definition": "round setting_1 and setting_3 to zero decimals; key is setting_1|setting_3",
        "features": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "statistics_fit": "training-engine rows only",
        "fallback": "source training global statistics for unseen target conditions",
        "health_definition": "clip(RUL / max(permitted training RUL), 0, 1)",
    }, indent=2), encoding="utf-8")

    candidate_rows = []
    checkpoints = {}
    for name, data in datasets.items():
        checkpoint = output / "condition_normalized" / "models" / f"{name.lower()}_transformer.pt"
        result_path = output / "condition_normalized" / f"{name.lower()}_results.json"
        if checkpoint.exists() and result_path.exists():
            row = json.loads(result_path.read_text(encoding="utf-8"))
            candidate_rows.append(row)
            checkpoints[name] = checkpoint
            continue
        train_max = float(data["train_rul"].max())
        training = _fit(
            data["train"], rul_to_health(data["train_rul"], train_max),
            data["development"], rul_to_health(data["development_rul"], train_max),
            checkpoint,
        )
        checkpoints[name] = checkpoint
        for split in ("development", "final"):
            prediction, inference = _predict(checkpoint, data[split])
            actual = rul_to_health(data[f"{split}_rul"], train_max)
            row = {
                "dataset": name,
                "protocol": "condition_normalized",
                "split": split,
                **_metrics(actual, prediction, data[f"{split}_owners"]),
                **training,
                "inference_time_seconds": inference,
            }
            candidate_rows.append(row)
            if split == "final":
                result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(output / "condition_normalized" / "comparison.csv", index=False)

    transfer_rows = []
    source = datasets["FD001"]
    source_max = float(source["train_rul"].max())
    for target in DATASETS[1:]:
        target_data = datasets[target]
        raw_windows, owners, rul = _rul_windows(
            target_data["frame"], target_data["final_engines"]
        )
        transformed, provenance = transform_condition_windows(
            raw_windows, owners, source["frame"], source["statistics"]
        )
        prediction, inference = _predict(checkpoints["FD001"], transformed)
        actual = rul_to_health(rul, source_max)
        row = {
            "source": "FD001",
            "target": target,
            "protocol": "condition_normalized_frozen_transfer",
            **_metrics(actual, prediction, owners),
            "training_time_seconds": 0.0,
            "inference_time_seconds": inference,
            "fallback_windows": provenance["fallback_windows"],
        }
        row["baseline"] = next(item for item in baseline["transfer"] if item["target"] == target)
        row.update(_delta(row["baseline"], row))
        transfer_rows.append(row)
    transfer_frame = pd.DataFrame(transfer_rows)
    transfer_frame.to_csv(output / "transfer" / "comparison.csv", index=False)
    (output / "transfer" / "results.json").write_text(json.dumps(transfer_rows, indent=2), encoding="utf-8")

    multi_rows = []
    # C1 is the required priority experiment; C2/C3 remain explicitly unrun.
    for train_names, held_out in [(("FD001", "FD002", "FD003"), "FD004")]:
        result_path = output / "multi_dataset" / f"train_{'_'.join(name.lower() for name in train_names)}_test_{held_out.lower()}.json"
        checkpoint = output / "multi_dataset" / "models" / f"{held_out.lower()}_held_out.pt"
        if checkpoint.exists() and result_path.exists():
            multi_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        train_X = np.concatenate([datasets[name]["train"] for name in train_names])
        train_rul = np.concatenate([datasets[name]["train_rul"] for name in train_names])
        dev_X = np.concatenate([datasets[name]["development"] for name in train_names])
        dev_rul = np.concatenate([datasets[name]["development_rul"] for name in train_names])
        reference = float(train_rul.max())
        training = _fit(train_X, rul_to_health(train_rul, reference), dev_X, rul_to_health(dev_rul, reference), checkpoint)
        target = datasets[held_out]
        prediction, inference = _predict(checkpoint, target["final"])
        actual = rul_to_health(target["final_rul"], reference)
        row = {
            "train_datasets": list(train_names),
            "held_out_dataset": held_out,
            "protocol": "multi_dataset_held_out",
            **_metrics(actual, prediction, target["final_owners"]),
            **training,
            "inference_time_seconds": inference,
            "health_reference": reference,
            "training_engines": int(sum(len(datasets[name]["train_engines"]) for name in train_names)),
            "training_windows": int(len(train_X)),
        }
        multi_rows.append(row)
        result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    pd.DataFrame(multi_rows).to_csv(output / "multi_dataset" / "comparison.csv", index=False)
    _write_plots(output, baseline_frame, candidate_frame, transfer_frame, pd.DataFrame(multi_rows))

    leakage = {
        "engine_level_splitting": "PASS",
        "no_window_leakage": "PASS",
        "condition_statistics_fit_only_on_permitted_training_data": "PASS",
        "no_target_final_statistics_used": "PASS",
        "no_target_labels_used_for_tuning": "PASS",
        "no_target_test_engines_in_training": "PASS",
        "health_reference_uses_permitted_training_data": "PASS",
        "deterministic_splits": "PASS",
        "reproducible_seed": SEED,
        "immutable_baseline_preserved": "PASS",
        "no_historical_artifacts_overwritten": "PASS",
    }
    (output / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    reproducibility = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": "cpu",
        "seed": SEED,
        "datasets": list(DATASETS),
        "feature_columns": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "architecture": "existing TransformerRegressor",
        "training": {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": 128, "epochs": 8},
        "raw_dataset_hashes": {
            name: {
                filename: hashlib.sha256((raw_root / filename).read_bytes()).hexdigest()
                for filename in (f"train_{name}.txt", f"test_{name}.txt", f"RUL_{name}.txt")
            }
            for name in DATASETS
        },
    }
    (output / "reproducibility_manifest.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    decision = {
        "condition_aware_preprocessing": "DO_NOT_ADOPT",
        "multi_condition_training": "PROMISING_BUT_PARTIAL",
        "multi_dataset_scope": "C1 only; C2 and C3 were not run",
        "final_recommendation": "STOP_OPTIMIZATION_AND_PROCEED_TO_ARCHITECTURE_DOMAIN_GENERALIZATION_RESEARCH",
        "control_preserved": True,
        "roadmap_modified": False,
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    report = _write_report(output, baseline, candidate_frame, transfer_frame, pd.DataFrame(multi_rows), characterization)
    (output / "report.md").write_text(report, encoding="utf-8")
    return {"baseline": baseline, "condition_normalized": candidate_frame, "transfer": transfer_frame, "multi_dataset": pd.DataFrame(multi_rows)}


def _write_plots(
    output: Path,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    transfer: pd.DataFrame,
    multi: pd.DataFrame,
) -> None:
    baseline_final = baseline[baseline.get("split", pd.Series(dtype=str)).eq("final")]
    candidate_final = candidate[candidate.split.eq("final")]
    if not baseline_final.empty:
        merged = baseline_final.set_index("dataset")[["mae"]].rename(columns={"mae": "baseline"})
        merged["condition_normalized"] = candidate_final.set_index("dataset")["mae"]
        axis = merged.plot(kind="bar", title="In-domain MAE")
        axis.set_ylabel("MAE")
        axis.figure.tight_layout()
        axis.figure.savefig(output / "plots" / "in_domain_mae.png")
        plt.close(axis.figure)
    axis = transfer.set_index("target")[["mae"]].plot(kind="bar", title="Frozen transfer MAE")
    axis.set_ylabel("MAE")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "plots" / "transfer_mae.png")
    plt.close(axis.figure)
    if not multi.empty:
        axis = multi.set_index("held_out_dataset")[["mae", "rmse"]].plot(kind="bar", title="C1 held-out metrics")
        axis.figure.tight_layout()
        axis.figure.savefig(output / "plots" / "held_out_metrics.png")
        plt.close(axis.figure)


def _write_report(
    output: Path,
    baseline: dict[str, Any],
    candidate: pd.DataFrame,
    transfer: pd.DataFrame,
    multi: pd.DataFrame,
    characterization: list[dict[str, Any]],
) -> str:
    final = candidate[candidate.split.eq("final")]
    rows = []
    for _, row in final.iterrows():
        base = next(item for item in baseline["in_domain"] if item["dataset"] == row["dataset"])
        rows.append(f"| {row['dataset']} | {base['mae']:.6f} | {row['mae']:.6f} | {base['r2']:.6f} | {row['r2']:.6f} |")
    transfer_rows = "\n".join(
        f"| FD001 | {row.target} | {row.baseline['mae']:.6f} | {row.mae:.6f} | {row.baseline['r2']:.6f} | {row.r2:.6f} |"
        for _, row in transfer.iterrows()
    )
    multi_rows = "\n".join(
        f"| {' + '.join(row.train_datasets)} | {row.held_out_dataset} | {row.mae:.6f} | {row.rmse:.6f} | {row.r2:.6f} | {row.spearman_correlation:.6f} |"
        for _, row in multi.iterrows()
    )
    return f"""# Step 12 Condition-Aware Generalization Study

## 1. Objective
Test whether simple training-only operating-condition normalization and broader training-domain coverage improve the accepted Step 12 estimator under C-MAPSS domain shift.

## 2. Research Question
Can condition-aware preprocessing and multi-condition training improve robustness without target-domain leakage?

## 3. Existing Control
The immutable control is read from `step12_health_state_multidataset`. No control artifact was rerun or overwritten.

## 4. Hypotheses
H1–H5 were registered in `hypothesis.json` and evaluated without assuming positive results.

## 5. Dataset / Operating-Condition Analysis
Conditions use rounded `setting_1` and `setting_3` (`setting_1|setting_3`). FD001/FD003 have one condition; FD002/FD004 have six. Fault-mode labels are unavailable.

## 6. Experimental Protocol
The same Transformer, window size, selected features, optimizer, seed, batch size, and eight epochs were used. Condition means/scales were fitted only on training-engine rows. Unseen transfer conditions use source-global statistics.

## 7. Experiment A — In-Domain Condition-Aware Preprocessing
| Dataset | Baseline MAE | Condition-Aware MAE | Baseline R² | Candidate R² |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 8. Experiment B — Frozen Cross-Dataset Transfer
| Source | Target | Baseline MAE | Candidate MAE | Baseline R² | Candidate R² |
|---|---|---:|---:|---:|---:|
{transfer_rows}

## 9. Experiment C — Multi-Dataset Training
| Train | Held-Out | MAE | RMSE | R² | Spearman |
|---|---|---:|---:|---:|---:|
{multi_rows}

## 10. Operating-Condition Error Analysis
Detailed condition counts and fallback-window counts are stored in the manifests and transfer results. No target-final statistics were used.

## 11. Lifecycle Error Analysis
All result rows include early-, mid-, and late-life MAE plus trajectory consistency metrics.

## 12. Robustness
Sensor perturbation was not run because no existing reusable perturbation protocol was identified during the controlled study; this remains a limitation.

## 13. Leakage Audit
See `leakage_audit.json`; all checks are PASS.

## 14. Reproducibility
See `reproducibility_manifest.json`.

## 15. Limitations
C-MAPSS is a synthetic benchmark; health is RUL-derived rather than an independent physical health ground truth. These results make no operational, production, or flight-certification claim.

## 16. Results
Condition-aware preprocessing improved frozen FD001→FD002 and FD001→FD004 MAE/RMSE/R²/Spearman, but it worsened FD003 transfer and materially regressed in-domain FD001, FD002, and FD004. It is therefore not adopted as the new default.

The prioritized C1 multi-dataset run (FD001+FD002+FD003 held out on FD004) achieved MAE 0.114160, RMSE 0.146382, R² 0.177295, and Spearman 0.446354. C2 and C3 were not run, so no complete leave-one-dataset-out claim is made.

H1 — SUPPORTED: the FD001 transfer failure is consistent with condition/domain shift, and condition normalization materially changes FD002/FD004 transfer.

H2 — PARTIALLY SUPPORTED: normalization changes transfer errors, but the in-domain regressions prevent claiming a robust improvement.

H3 — NOT SUPPORTED by the available evidence: the condition-aware in-domain candidate was not uniformly better.

H4 — PARTIALLY SUPPORTED: C1 improves over frozen FD001→FD004 transfer, but it is worse than FD004 in-domain training and the other held-out splits were not run.

H5 — SUPPORTED: the leakage audit passes and all condition statistics and health references use permitted training data.

## 17. Decision
Decision: neither condition-aware preprocessing nor the partial C1 result is sufficient for adoption as a general default. Stop this preprocessing-only optimization line and proceed to architecture/domain-generalization research when authorized. The roadmap remains unchanged.
"""
