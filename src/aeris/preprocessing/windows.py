import numpy as np
import pandas as pd


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
):
    """
    Convert engine telemetry into temporal sequences.
    """

    # Store the generated feature windows, target values, and engine
    # identifiers so they remain aligned throughout sequence creation.
    X = []
    y = []
    engine_ids = []

    # Process each engine independently to preserve the temporal
    # structure of individual engine degradation trajectories.
    for engine_id, group in df.groupby("engine_id"):

        # Ensure telemetry observations are ordered chronologically
        # before constructing the temporal windows.
        group = group.sort_values("cycle")

        # Convert the selected telemetry features into numerical arrays
        # suitable for downstream sequence-model processing.
        features = group[feature_columns].to_numpy(
            dtype=np.float32
        )

        # Extract the target value associated with each telemetry cycle.
        targets = group[target_column].to_numpy(
            dtype=np.float32
        )

        # Skip trajectories that do not contain enough cycles to form
        # the requested temporal window.
        if len(group) < window_size:
            continue

        # Generate overlapping temporal windows across the engine's
        # complete observed trajectory.
        for end in range(
            window_size,
            len(group) + 1,
        ):

            # Calculate the starting position of the current window.
            start = end - window_size

            # Store the fixed-length sequence of telemetry features.
            X.append(
                features[start:end]
            )

            # Use the target at the final timestep as the prediction
            # target for the complete temporal sequence.
            y.append(
                targets[end - 1]
            )

            # Preserve the originating engine ID for later
            # engine-level analysis and leakage-safe evaluation.
            engine_ids.append(engine_id)

    # Convert the generated sequences, targets, and engine identifiers
    # into NumPy arrays for downstream machine-learning workflows.
    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(engine_ids),
    )