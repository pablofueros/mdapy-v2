"""Numerical reference for the Python/NumPy evolution planned for mdapy 2.2.

Source: working tree on ``perf-2.2-python-numpy``. Preparation preserves the
2.1 calculations. Selection reuses working buffers and compares squared
distances for real inputs; complex inputs retain NumPy's norm reduction.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import circstd

SOURCE = "perf-2.2-python-numpy"


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
    """Run incremental selection with reusable buffers and squared distances."""
    indices = [seed_index]
    selected = np.zeros(len(prepared.data), dtype=bool)
    selected[seed_index] = True
    nearest = np.full(len(prepared.data), np.inf)
    if n_clusters == 1:
        return prepared.data[indices], indices
    differences = np.empty_like(prepared.data)
    normalized = np.empty(
        prepared.data.shape,
        dtype=np.result_type(
            prepared.data.dtype, np.asarray(prepared.deviations).dtype
        ),
    )

    while len(indices) < n_clusters:
        np.subtract(prepared.data, prepared.data[indices[-1]], out=differences)
        for column in prepared.directional:
            differences[:, column] = (differences[:, column] + 180) % 360 - 180
        np.divide(differences, prepared.deviations, out=normalized)
        if np.iscomplexobj(normalized):
            distances = np.linalg.norm(normalized, axis=1)
        else:
            distances = np.einsum("ij,ij->i", normalized, normalized)
        np.minimum(nearest, distances, out=nearest)
        nearest[selected] = -np.inf
        next_index = int(np.argmax(nearest))
        indices.append(next_index)
        selected[next_index] = True

    return prepared.data[indices], indices
