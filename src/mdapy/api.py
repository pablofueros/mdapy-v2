"""Column-based API for selecting maximally dissimilar observations."""

from collections.abc import Callable
from functools import cached_property
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import circstd


class DataColumn:
    """A stable copy of one variable, optionally interpreted as degrees.

    Values must have a real integer or floating-point dtype. Their original
    dtype and directional interpretation are read-only through the public API,
    so the cached standard deviation remains consistent with the column.
    """

    def __init__(self, array: ArrayLike, *, directional: bool = False) -> None:
        values = np.asarray(array)
        if not isinstance(directional, bool):
            raise TypeError("directional must be a bool")
        if values.ndim != 1:
            raise ValueError("Each data column must be one-dimensional")
        if values.dtype.kind not in "iuf":
            raise TypeError("Data columns must have a real numeric dtype")
        if not np.isfinite(values).all():
            raise ValueError("Data columns must contain only finite values")
        self._values = values.copy()
        self._values.setflags(write=False)
        self._directional = directional

    @property
    def values(self) -> NDArray:
        """Return a read-only view backed by the column's protected copy."""
        return self._values.view()

    @property
    def directional(self) -> bool:
        """Whether this column represents angles in degrees."""
        return self._directional

    def __len__(self) -> int:
        return len(self._values)

    @cached_property
    def standard_deviation(self) -> float:
        values = self.values.astype(np.float64, copy=False)
        if self.directional:
            return float(circstd(values, high=360, low=0))
        return float(np.std(values))


_LEGACY_HINT = (
    "The tuple-based API is available from mdapy.legacy: "
    "from mdapy.legacy import max_diss_alg"
)


def _get_select_indices(
    engine: Literal["numpy", "numba"],
) -> Callable[..., list[int]]:
    """Load the internal selection function for the requested engine."""
    if engine == "numpy":
        from .core import _select_indices

        return _select_indices

    try:
        from .numba import _select_indices
    except ModuleNotFoundError as error:
        if error.name != "numba":
            raise
        raise ImportError(
            "The Numba engine requires the optional dependency. Install it "
            'with: pip install "mdapy-v2[numba]"'
        ) from error

    return _select_indices


def max_diss_alg(
    *columns: DataColumn,
    n_clusters: int,
    seed_index: int = 0,
    engine: Literal["numpy", "numba"] = "numpy",
    **kwargs: object,
) -> tuple[NDArray, list[int]]:
    """Select maximally dissimilar observations.

    Parameters
    ----------
    columns : DataColumn
        Columns describing each variable. Directional columns represent angles
        in degrees with a period of 360.
    n_clusters : int
        Number of observations to select.
    seed_index : int, optional
        Index of the first selected observation, by default 0.
    engine : {"numpy", "numba"}, optional
        Numerical engine used for the selection, by default "numpy". The
        "numba" engine requires the optional ``numba`` installation extra and
        is intended for large or repeated selections.

    Returns
    -------
    selected_data : NDArray
        Selected observations in selection order.
    selected_indices : list[int]
        Indices of the selected observations in the original data.

    Notes
    -----
    Selection calculations use ``float64``. The returned observations retain
    the common NumPy dtype of the input columns.
    """

    if "show_progress" in kwargs:
        raise TypeError("show_progress has been removed")
    if "data_tuple" in kwargs or "dir_indices" in kwargs:
        raise TypeError(_LEGACY_HINT)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"Unexpected keyword argument: {unexpected}")

    if columns and not all(isinstance(col, DataColumn) for col in columns):
        raise TypeError(f"Expected DataColumn arguments. {_LEGACY_HINT}")
    if not columns:
        raise ValueError("At least one DataColumn is required")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, Integral):
        raise TypeError("n_clusters must be an integer")
    if isinstance(seed_index, bool) or not isinstance(seed_index, Integral):
        raise TypeError("seed_index must be an integer")
    if engine not in ("numpy", "numba"):
        raise ValueError("engine must be either 'numpy' or 'numba'")

    n_points = len(columns[0])
    if any(len(col) != n_points for col in columns):
        raise ValueError("All data columns must have the same length")
    if not 1 <= n_clusters <= n_points:
        raise ValueError("n_clusters must be between 1 and the number of rows")
    if not 0 <= seed_index < n_points:
        raise ValueError("seed_index is out of bounds")

    deviations = np.asarray([col.standard_deviation for col in columns])
    for index, deviation in enumerate(deviations):
        if not np.isfinite(deviation) or deviation <= 0:
            raise ValueError(f"Column {index} has zero or undefined deviation")

    data = np.column_stack(tuple(col.values for col in columns))
    calculation_data = data.astype(np.float64, copy=False)
    directional = [i for i, col in enumerate(columns) if col.directional]

    select_indices = _get_select_indices(engine)
    indices = select_indices(
        calculation_data,
        deviations,
        directional,
        n_clusters,
        seed_index,
    )

    return data[indices], indices
