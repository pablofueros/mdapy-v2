"""Compare NumPy and Numba on a workload where seconds can be saved.

The first Numba call compiles the kernel or loads it from the disk cache. The
second call reuses the compiled specialization. Timings depend on the machine;
the exact equality checks are the relevant correctness result.
"""

from time import perf_counter
from typing import Literal

import numpy as np

from mdapy import DataColumn, max_diss_alg

ROW_COUNT = 121_968
COLUMN_COUNT = 3
SELECTION_COUNT = 5_000
RANDOM_SEED = 20261020

rng = np.random.default_rng(RANDOM_SEED)
columns = tuple(DataColumn(rng.normal(size=ROW_COUNT)) for _ in range(COLUMN_COUNT))


def timed_selection(
    engine: Literal["numpy", "numba"],
) -> tuple[np.ndarray, list[int], float]:
    """Run one public-API selection and return its elapsed time."""
    started = perf_counter()
    selected, indices = max_diss_alg(
        *columns,
        n_clusters=SELECTION_COUNT,
        seed_index=0,
        engine=engine,
    )
    return selected, indices, perf_counter() - started


numpy_selected, numpy_indices, numpy_seconds = timed_selection("numpy")
numba_first_selected, numba_first_indices, numba_first_seconds = timed_selection(
    "numba"
)
numba_hot_selected, numba_hot_indices, numba_hot_seconds = timed_selection("numba")

assert numba_first_indices == numba_hot_indices == numpy_indices
np.testing.assert_array_equal(numba_first_selected, numpy_selected)
np.testing.assert_array_equal(numba_hot_selected, numpy_selected)

print(f"Observations: {ROW_COUNT:,}")
print(f"Variables: {COLUMN_COUNT}")
print(f"Selections: {SELECTION_COUNT:,}")
print()
print(f"NumPy:                       {numpy_seconds:.3f} s")
print(f"Numba, first call:           {numba_first_seconds:.3f} s")
print(f"Numba, repeated call:        {numba_hot_seconds:.3f} s")
print()
print(f"Saved on first call:         {numpy_seconds - numba_first_seconds:.3f} s")
print(f"Saved on repeated call:      {numpy_seconds - numba_hot_seconds:.3f} s")
print(f"Repeated-call speedup:       {numpy_seconds / numba_hot_seconds:.2f}x")
print("Selected indices match exactly.")
