"""Optional Numba core for selecting maximally dissimilar observations."""

from numbers import Integral

import numpy as np
from numba import (
    get_num_threads,
    njit,
    prange,
    set_num_threads,
)
from numpy.typing import NDArray


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


def _select_indices(
    data: NDArray,
    deviations: NDArray,
    dir_indices: list[int] | None,
    n_clusters: int,
    seed_index: int,
    *,
    show_progress: bool = False,
    workers: int | None = None,
) -> list[int]:
    """Adapt the compiled kernel to the internal NumPy core interface."""
    if show_progress:
        raise NotImplementedError("The Numba engine does not support progress output")
    if workers is not None:
        if isinstance(workers, bool) or not isinstance(workers, Integral):
            raise TypeError("workers must be an integer or None")
        if workers < 1:
            raise ValueError("workers must be a positive integer")

    directional = np.zeros(data.shape[1], dtype=np.bool_)
    directional[dir_indices or []] = True

    previous_workers = None
    if workers is not None:
        previous_workers = get_num_threads()
        set_num_threads(int(workers))

    try:
        indices = _select_indices_kernel(
            data,
            deviations,
            directional,
            n_clusters,
            seed_index,
        )
    finally:
        if previous_workers is not None:
            set_num_threads(previous_workers)

    return indices.tolist()
