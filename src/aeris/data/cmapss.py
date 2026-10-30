from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def load_cmapss_file(path: str | Path) -> pd.DataFrame:
    """
    Load a NASA C-MAPSS whitespace-delimited telemetry file.

    Parameters
    ----------
    path:
        Path to the raw C-MAPSS file.

    Returns
    -------
    pandas.DataFrame
        Standardized telemetry dataframe.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"C-MAPSS file not found: {path}"
        )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    return df


@dataclass(frozen=True)
class CMapssDataset:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul: np.ndarray

    @property
    def sensor_columns(self) -> list[str]:
        return SENSOR_COLUMNS.copy()

    @property
    def operating_condition_columns(self) -> list[str]:
        return ["setting_1", "setting_2", "setting_3"]

    def get_engine_ids(self) -> np.ndarray:
        return self.train["engine_id"].drop_duplicates().to_numpy()

    def get_metadata(self) -> dict[str, object]:
        return {"dataset": self.name, "sensor_columns": self.sensor_columns,
                "operating_condition_columns": self.operating_condition_columns,
                "train_engines": int(self.train["engine_id"].nunique()),
                "test_engines": int(self.test["engine_id"].nunique())}


def load_cmapss(root: str | Path, dataset: str) -> CMapssDataset:
    """Load one C-MAPSS variant and preserve settings and sensors."""
    dataset = dataset.upper()
    if dataset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    root = Path(root)
    train = load_cmapss_file(root / f"train_{dataset}.txt")
    test = load_cmapss_file(root / f"test_{dataset}.txt")
    rul = np.loadtxt(root / f"RUL_{dataset}.txt", dtype=float, ndmin=1)
    if len(rul) != test["engine_id"].nunique():
        raise ValueError(f"RUL count does not match test engines for {dataset}")
    return CMapssDataset(dataset, train, test, rul)


def inspect_dataset(dataset: CMapssDataset) -> dict[str, object]:
    train = dataset.train
    monotonic = all(group["cycle"].is_monotonic_increasing for _, group in train.groupby("engine_id"))
    return {**dataset.get_metadata(), "train_rows": int(len(train)),
            "test_rows": int(len(dataset.test)), "train_missing_values": int(train.isna().sum().sum()),
            "test_missing_values": int(dataset.test.isna().sum().sum()),
            "duplicate_train_rows": int(train.duplicated().sum()),
            "train_cycles_monotonic": bool(monotonic),
            "operating_condition_count": int(train[dataset.operating_condition_columns].drop_duplicates().shape[0]),
            "engine_cycle_min": int(train["cycle"].min()), "engine_cycle_max": int(train["cycle"].max())}


def split_engines(engine_ids: Iterable[int], seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    ids = np.asarray(list(engine_ids), dtype=int)
    if len(ids) < 3:
        raise ValueError("At least three engines are required")
    shuffled = np.random.default_rng(seed).permutation(ids)
    train_end = max(1, int(len(ids) * 0.6))
    calibration_end = min(max(train_end + 1, int(len(ids) * 0.8)), len(ids) - 1)
    return shuffled[:train_end].tolist(), shuffled[train_end:calibration_end].tolist(), shuffled[calibration_end:].tolist()


def make_windows(frame: pd.DataFrame, engine_ids: Iterable[int], sensor_columns: list[str],
                 window_size: int = 30) -> tuple[np.ndarray, np.ndarray]:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    selected = {int(value) for value in engine_ids}
    windows, owners = [], []
    for engine_id, group in frame.groupby("engine_id", sort=True):
        if int(engine_id) not in selected:
            continue
        values = group.sort_values("cycle")[sensor_columns].to_numpy(dtype=np.float32)
        for end in range(window_size, len(values) + 1):
            windows.append(values[end - window_size:end])
            owners.append(int(engine_id))
    if not windows:
        raise ValueError("No windows were generated")
    return np.asarray(windows, dtype=np.float32), np.asarray(owners, dtype=int)


def fit_sensor_scaler(windows: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(windows, dtype=np.float64).reshape(-1, windows.shape[-1])
    return {"mean": values.mean(axis=0), "scale": np.maximum(values.std(axis=0), 1e-8)}


def transform_windows(windows: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((np.asarray(windows, dtype=np.float64) - scaler["mean"]) / scaler["scale"]).astype(np.float32)