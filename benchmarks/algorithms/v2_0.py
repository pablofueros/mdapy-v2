"""Numerical reference for mdapy 2.0.0.

Source: Git tag v2.0.0. The common benchmark interface replaces only API
validation and progress output; conversion, deviations and selection preserve
the released numerical path.
"""

from functools import cache

import numpy as np
from numpy.typing import NDArray
from scipy.stats import circstd

SOURCE = "v2.0.0"


def deviation(column: NDArray, directional: bool) -> float:
    if directional:
        return float(circstd(column, high=360, low=0))
    return float(np.std(column))


class DataMatrix:
    """Prepared float32 matrix and historical pair-distance cache."""

    def __init__(self, columns: tuple[NDArray, ...], directional: list[int]) -> None:
        converted = tuple(np.asarray(column, dtype=np.float32) for column in columns)
        self.data = np.array(converted, dtype=np.float32).T
        self.deviations = [
            deviation(column, index in directional)
            for index, column in enumerate(converted)
        ]
        self.directional = directional

    @cache  # noqa: B019 -- This cache is part of the 2.0 algorithm.
    def distance(self, first: int, second: int) -> float:
        difference = np.zeros_like(self.data[first])
        for column, (left, right) in enumerate(
            zip(self.data[first], self.data[second], strict=True)
        ):
            delta = (
                ((left - right) + 180) % 360 - 180
                if column in self.directional
                else left - right
            )
            difference[column] = delta / (np.sqrt(2) * self.deviations[column])
        return float(np.linalg.norm(difference))


def prepare(columns: tuple[NDArray, ...], directional: list[int]) -> DataMatrix:
    """Prepare inputs as the 2.0 public call did."""
    converted = tuple(np.asarray(column, dtype=np.float32) for column in columns)
    for index, column in enumerate(converted):
        if not np.isfinite(column).all():
            raise ValueError("Columns must contain only finite values")
        checked = deviation(column, index in directional)
        if not np.isfinite(checked) or checked <= 0:
            raise ValueError("Columns must have a finite, positive deviation")
    return DataMatrix(converted, directional)


def select(
    prepared: DataMatrix, n_clusters: int, seed_index: int
) -> tuple[NDArray, list[int]]:
    """Run the cached, subset-scanning selection from 2.0."""
    indices = [seed_index]
    while len(indices) < n_clusters:
        maximum = -1.0
        next_index = -1
        for candidate in range(len(prepared.data)):
            if candidate in indices:
                continue
            nearest = min(prepared.distance(candidate, chosen) for chosen in indices)
            if nearest > maximum:
                maximum = nearest
                next_index = candidate
        indices.append(next_index)
    return prepared.data[indices], indices
