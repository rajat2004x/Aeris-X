from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_engine(
    df: pd.DataFrame,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split telemetry data by engine ID.

    This prevents telemetry from the same engine appearing
    in both training and validation sets.
    """

    engine_ids = df["engine_id"].unique()

    train_engines, validation_engines = train_test_split(
        engine_ids,
        test_size=validation_size,
        random_state=random_state,
    )

    train_df = df[
        df["engine_id"].isin(train_engines)
    ].copy()

    validation_df = df[
        df["engine_id"].isin(validation_engines)
    ].copy()

    return train_df, validation_df