"""Tuple-based compatibility API for selecting maximally dissimilar observations."""

from numbers import Integral

from numpy.typing import ArrayLike, NDArray

from .api import DataColumn
from .api import max_diss_alg as _current_max_diss_alg


def max_diss_alg(
    data_tuple: tuple[ArrayLike, ...],
    n_clusters: int,
    seed_index: int = 0,
    dir_indices: list[int] | None = None,
    show_progress: bool = False,
) -> tuple[NDArray, list[int]]:
    """Select maximally dissimilar observations using the tuple-based API.

    This compatibility wrapper converts the input arrays into `DataColumn`
    objects and delegates the selection to the current API.

    Parameters
    ----------
    data_tuple : tuple[ArrayLike, ...]
        One-dimensional arrays describing each variable.
    n_clusters : int
        Number of observations to select.
    seed_index : int, optional
        Index of the first selected observation, by default 0.
    dir_indices : list[int] or None, optional
        Indices of arrays containing angles in degrees, by default None.
    show_progress : bool, optional
        Accepted for backward compatibility and currently ignored, by default
        False.

    Returns
    -------
    selected_data : NDArray
        Selected observations in selection order.
    selected_indices : list[int]
        Indices of the selected observations in the original data.
    """

    directional = set(dir_indices or [])
    for index in directional:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("dir_indices must contain only integers")
        if not 0 <= index < len(data_tuple):
            raise ValueError("dir_indices contains an out-of-bounds index")

    columns = tuple(
        DataColumn(array, directional=index in directional)
        for index, array in enumerate(data_tuple)
    )

    return _current_max_diss_alg(
        *columns,
        n_clusters=n_clusters,
        seed_index=seed_index,
        show_progress=show_progress,
    )
