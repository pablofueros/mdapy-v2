"""Check the user experience without the optional Numba dependency."""

import numpy as np

from mdapy import DataColumn, max_diss_alg

selected, indices = max_diss_alg(DataColumn([0, 2, 5, 9]), n_clusters=3)
assert indices == [0, 3, 2]
np.testing.assert_array_equal(selected[:, 0], [0, 9, 5])

try:
    max_diss_alg(DataColumn([0, 1]), n_clusters=1, engine="numba")
except ImportError as error:
    assert 'pip install "mdapy-v2[numba]"' in str(error)
else:
    raise AssertionError("The base installation unexpectedly includes Numba")
