import pandas as pd


def generate_quality_report(
    df: pd.DataFrame,
) -> dict:
    """
    Generate a basic telemetry data-quality report.
    """

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith("sensor_")
    ]

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "engines": df["engine_id"].nunique(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_engine_cycles": int(
            df.duplicated(
                subset=["engine_id", "cycle"]
            ).sum()
        ),
        "sensor_columns": len(sensor_columns),
    }

    return report