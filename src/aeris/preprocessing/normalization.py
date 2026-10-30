from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_sensor_scaler(
    df: pd.DataFrame,
    sensor_columns: List[str],
) -> StandardScaler:
    """
    Fit a StandardScaler using training telemetry only.
    """

    scaler = StandardScaler()

    scaler.fit(df[sensor_columns])

    return scaler


def transform_sensor_data(
    df: pd.DataFrame,
    sensor_columns: List[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Transform telemetry using an already-fitted scaler.
    """

    result = df.copy()

    result[sensor_columns] = scaler.transform(
        result[sensor_columns]
    )

    return result


def save_scaler(
    scaler: StandardScaler,
    path: str | Path,
) -> None:
    """
    Save fitted scaler to disk.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(scaler, path)


def load_scaler(
    path: str | Path,
) -> StandardScaler:
    """
    Load a previously fitted scaler.
    """

    return joblib.load(path)