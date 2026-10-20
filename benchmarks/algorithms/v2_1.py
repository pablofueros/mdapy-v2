"""Numerical reference for mdapy 2.1.0.

Source: Git tag v2.1.0. The common benchmark interface separates preparation
from selection without changing the released calculations.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import circstd

SOURCE = "v2.1.0"


@dataclass
class PreparedData:
    data: NDArray
    deviations: list[float]
    directional: list[int]


def prepare(columns: tuple[NDArray, ...], directional: list[int]) -> PreparedData:
    """Copy columns, calculate deviations and build their common NumPy dtype."""
    copied = tuple(np.asarray(column).copy() for column in columns)
    if any(not np.isfinite(column).all() for column in copied):
        raise ValueError("Columns must contain only finite values")
    deviations = [
        float(circstd(column, high=360, low=0))
        if index in directional
        else float(np.std(column))
        for index, column in enumerate(copied)
    ]
    if any(not np.isfinite(value) or value <= 0 for value in deviations):
        raise ValueError("Columns must have a finite, positive deviation")
    return PreparedData(np.column_stack(copied), deviations, directional)


def select(
    prepared: PreparedData, n_clusters: int, seed_index: int
) -> tuple[NDArray, list[int]]:
    """Run the incremental minimum-distance selection from 2.1."""
    indices = [seed_index]
    selected = np.zeros(len(prepared.data), dtype=bool)
    selected[seed_index] = True
    nearest = np.full(len(prepared.data), np.inf)

    while len(indices) < n_clusters:
        differences = prepared.data - prepared.data[indices[-1]]
        for column in prepared.directional:
            differences[:, column] = (differences[:, column] + 180) % 360 - 180
        distances = np.linalg.norm(differences / prepared.deviations, axis=1)
        np.minimum(nearest, distances, out=nearest)
        nearest[selected] = -np.inf
        next_index = int(np.argmax(nearest))
        indices.append(next_index)
        selected[next_index] = True

    return prepared.data[indices], indices
