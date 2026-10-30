import pandas as pd


def add_rul_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) labels
    to a training telemetry dataframe.
    """

    # Create a copy so that RUL generation does not modify the
    # original telemetry dataset unexpectedly.
    result = df.copy()

    # Determine the final observed cycle for each engine.
    # This provides the reference point for calculating RUL.
    max_cycles = (
        result.groupby("engine_id")["cycle"]
        .transform("max")
    )

    # Calculate the remaining number of cycles for every
    # telemetry observation.
    result["rul"] = max_cycles - result["cycle"]

    return result


def add_health_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add normalized health-state labels to telemetry data.
    """

    # Create a copy so that health-label generation does not
    # modify the input DataFrame in place.
    result = df.copy()

    # Determine the maximum RUL observed for each engine.
    # This represents the initial RUL used to normalize health.
    max_rul = (
        result.groupby("engine_id")["rul"]
        .transform("max")
    )

    # Normalize RUL to create a continuous health-state label.
    # Values near 1 represent the beginning of the trajectory,
    # while values near 0 represent the end of the trajectory.
    result["health"] = (
        result["rul"] / max_rul
    )

    return result