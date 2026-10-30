from typing import List

import pandas as pd


def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """
    Return all telemetry sensor columns.
    """

    # Identify sensor measurements using the standardized naming
    # convention established by the AERIS-X C-MAPSS data loader.
    # This avoids hard-coding the complete list of sensor names.
    return [
        column
        for column in df.columns
        if column.startswith("sensor_")
    ]


def find_constant_sensors(
    df: pd.DataFrame,
    tolerance: float = 1e-8,
) -> List[str]:
    """
    Identify sensors with effectively zero variance.
    """

    # Retrieve the telemetry columns that should be evaluated
    # for meaningful variation across the dataset.
    sensor_columns = get_sensor_columns(df)

    # Calculate the standard deviation of each sensor.
    # Sensors with negligible variation provide little changing
    # information for degradation modeling and require further review.
    std = df[sensor_columns].std()

    # Return sensors whose variability falls within the configured
    # tolerance. These sensors are candidates for exclusion from
    # the initial prognostics feature set, but are not automatically
    # deleted because they may still have value for other AERIS-X
    # tasks such as anomaly detection or fault diagnosis.
    return std[std <= tolerance].index.tolist()


def select_variable_sensors(
    df: pd.DataFrame,
) -> List[str]:
    """
    Return sensors that exhibit measurable variation.
    """

    # Retrieve all telemetry sensors identified by the standardized
    # AERIS-X sensor naming convention.
    sensors = get_sensor_columns(df)

    # Measure the variability of each sensor across the dataset.
    # This provides the basis for separating informative signals
    # from effectively constant telemetry measurements.
    std = df[sensors].std()

    # Keep sensors with measurable variation as candidates for the
    # initial feature set used by the AERIS-X prognostics pipeline.
    return std[std > 1e-8].index.tolist()