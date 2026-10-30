import pandas as pd


REQUIRED_COLUMNS = {
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
}


def validate_telemetry(df: pd.DataFrame) -> None:
    """
    Validate basic structural integrity of telemetry data.
    """

    if df.empty:
        raise ValueError("Telemetry dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["engine_id"].isna().any():
        raise ValueError(
            "Telemetry contains missing engine IDs."
        )

    if df["cycle"].isna().any():
        raise ValueError(
            "Telemetry contains missing cycle values."
        )

    if (df["cycle"] <= 0).any():
        raise ValueError(
            "Cycle values must be positive."
        )

    if df.duplicated(
        subset=["engine_id", "cycle"]
    ).any():
        raise ValueError(
            "Duplicate engine/cycle records detected."
        )