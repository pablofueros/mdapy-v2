"""Internal numerical core shared by the column and tuple APIs."""

import numpy as np
from numpy.typing import NDArray


def _squared_row_norms(values: NDArray) -> NDArray:
    """Return the squared Euclidean norm of each row."""
    # "ij,ij->i" multiplies each row by itself and sums its columns.
    return np.einsum("ij,ij->i", values, values)


def _squared_distances_to_point(
    data: NDArray,
    point_index: int,
    deviations: NDArray,
    dir_indices: list[int] | None,
    differences: NDArray,
    normalized: NDArray,
) -> NDArray:
    """Calculate squared distances using caller-owned working buffers."""
    np.subtract(data, data[point_index], out=differences)
    for column in dir_indices or []:
        differences[:, column] = (differences[:, column] + 180) % 360 - 180
    np.divide(differences, deviations, out=normalized)
    return _squared_row_norms(normalized)


def _select_indices(
    data: NDArray,
    deviations: NDArray,
    dir_indices: list[int] | None,
    n_clusters: int,
    seed_index: int,
    *,
    show_progress: bool = False,
) -> list[int]:
    """Update nearest distances only against the last selected observation."""
    indices = [seed_index]
    if n_clusters == 1:
        return indices

    selected = np.zeros(len(data), dtype=bool)
    selected[seed_index] = True
    nearest = np.full(len(data), np.inf)
    differences = np.empty_like(data)
    normalized = np.empty_like(data, dtype=np.result_type(data.dtype, deviations.dtype))

    while len(indices) < n_clusters:
        distances = _squared_distances_to_point(
            data,
            indices[-1],
            deviations,
            dir_indices,
            differences,
            normalized,
        )
        np.minimum(nearest, distances, out=nearest)
        # Negative infinity excludes selected rows even when all others coincide.
        nearest[selected] = -np.inf
        # argmax returns the first input index for an exact tie, as historically.
        next_index = int(np.argmax(nearest))
        if show_progress:
            for point_index in range(len(data)):
                if selected[point_index]:
                    continue
                progress = (point_index + 1) / len(data) * 100
                print(
                    f"Finding cluster nº{len(indices) + 1} --> {progress:.2f}%",
                    end="\r",
                )
        indices.append(next_index)
        selected[next_index] = True

    return indices
