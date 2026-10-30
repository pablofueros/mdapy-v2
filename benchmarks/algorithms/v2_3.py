"""Numerical reference for the optional Numba engine in mdapy 2.3.

The public API prepares all calculations as float64 and preserves the common
input dtype only for the returned observations. The compiled kernel updates
row distances in parallel and leaves Numba's thread configuration untouched.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.stats import circstd

SOURCE = "mdapy 2.3 Numba engine"


@dataclass
class PreparedData:
    data: NDArray
    calculation_data: NDArray
    deviations: NDArray
    directional: NDArray[np.bool_]


def prepare(columns: tuple[NDArray, ...], directional: list[int]) -> PreparedData:
    """Reproduce the 2.3 public API's numerical preparation."""
    copied = tuple(np.asarray(column).copy() for column in columns)
    if any(not np.isfinite(column).all() for column in copied):
        raise ValueError("Columns must contain only finite values")

    values64 = tuple(column.astype(np.float64, copy=False) for column in copied)
    directional_set = set(directional)
    deviations = np.asarray(
        [
            float(circstd(column, high=360, low=0))
            if index in directional_set
            else float(np.std(column))
            for index, column in enumerate(values64)
        ]
    )
    if any(not np.isfinite(value) or value <= 0 for value in deviations):
        raise ValueError("Columns must have a finite, positive deviation")

    data = np.column_stack(copied)
    directional_flags = np.zeros(data.shape[1], dtype=np.bool_)
    directional_flags[directional] = True
    return PreparedData(
        data,
        data.astype(np.float64, copy=False),
        deviations,
        directional_flags,
    )


@njit(cache=True, parallel=True)
def _select_indices_kernel(
    data: NDArray,
    deviations: NDArray,
    directional: NDArray[np.bool_],
    n_clusters: int,
    seed_index: int,
) -> NDArray[np.int64]:
    """Select indices while updating row distances in parallel."""
    row_count, column_count = data.shape
    indices = np.empty(n_clusters, dtype=np.int64)
    indices[0] = seed_index
    if n_clusters == 1:
        return indices

    selected = np.zeros(row_count, dtype=np.bool_)
    selected[seed_index] = True
    nearest = np.full(row_count, np.inf)

    for selection_position in range(1, n_clusters):
        last_index = indices[selection_position - 1]
        for row_index in prange(row_count):
            distance = 0.0
            for column_index in range(column_count):
                difference = (
                    data[row_index, column_index] - data[last_index, column_index]
                )
                if directional[column_index]:
                    difference = (difference + 180) % 360 - 180
                normalized = difference / deviations[column_index]
                distance += normalized * normalized
            nearest[row_index] = min(nearest[row_index], distance)

        next_index = 0
        greatest_distance = -np.inf
        for row_index in range(row_count):
            if selected[row_index]:
                nearest[row_index] = -np.inf
            elif nearest[row_index] > greatest_distance:
                greatest_distance = nearest[row_index]
                next_index = row_index

        indices[selection_position] = next_index
        selected[next_index] = True

    return indices


def select(
    prepared: PreparedData, n_clusters: int, seed_index: int
) -> tuple[NDArray, list[int]]:
    """Run the compiled 2.3 selection and retain the input dtype in output."""
    indices = _select_indices_kernel(
        prepared.calculation_data,
        prepared.deviations,
        prepared.directional,
        n_clusters,
        seed_index,
    ).tolist()
    return prepared.data[indices], indices
