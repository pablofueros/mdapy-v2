import numpy as np
import pytest
from scipy.stats import circstd

from mdapy import DataColumn, max_diss_alg
from mdapy.core import _select_indices as select_indices_numpy

numba = pytest.importorskip("numba")
numba_core = pytest.importorskip("mdapy.numba")

available_workers = tuple(
    workers
    for workers in (None, 1, 3, 4)
    if workers is None or workers <= numba.config.NUMBA_NUM_THREADS
)


def deviations_for(data, directional):
    return np.asarray(
        [
            circstd(data[:, index], high=360, low=0)
            if index in directional
            else np.std(data[:, index])
            for index in range(data.shape[1])
        ]
    )


@pytest.mark.parametrize("workers", available_workers)
@pytest.mark.parametrize(
    ("data", "directional", "n_clusters", "seed_index"),
    [
        (np.asarray([[-2], [0], [2]]), [], 3, 1),
        (np.asarray([[0], [0], [2], [2]]), [], 4, 0),
        (
            np.asarray(
                [[0, 350], [0, 10], [4, 180], [4, 90], [0, 350]],
                dtype=np.float64,
            ),
            [1],
            5,
            0,
        ),
        (
            np.asarray(
                [[0, -370], [1, 350], [2, 730], [3, 180]],
                dtype=np.float64,
            ),
            [1],
            4,
            0,
        ),
        (np.asarray([[0], [2], [5], [9]]), [], 3, 2),
    ],
    ids=["ties", "duplicates", "directional", "wrapped-angles", "seed"],
)
def test_numba_matches_numpy_reference_cases(
    data, directional, n_clusters, seed_index, workers
):
    deviations = deviations_for(data, directional)

    expected = select_indices_numpy(
        data,
        deviations,
        directional,
        n_clusters,
        seed_index,
    )
    actual = numba_core._select_indices(
        data,
        deviations,
        directional,
        n_clusters,
        seed_index,
        workers=workers,
    )

    assert actual == expected


@pytest.mark.parametrize("workers", available_workers)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numba_preserves_nearby_distance_order(dtype, workers):
    data = np.column_stack(
        (
            np.asarray([0, 1, 1, 0, 0, 0, 0, 0], dtype=dtype),
            np.asarray([0, 0, 0.000173, -0.5, 0.5, -0.5, 0.5, 0], dtype=dtype),
        )
    )
    deviations = deviations_for(data, [])

    expected = select_indices_numpy(data, deviations, [], 3, 0)
    actual = numba_core._select_indices(
        data,
        deviations,
        [],
        3,
        0,
        workers=workers,
    )

    assert actual == expected == [0, 2, 3]


def test_numba_restores_previous_worker_count():
    data = np.asarray([[0.0], [1.0], [2.0]])
    deviations = deviations_for(data, [])
    previous_workers = numba.get_num_threads()

    numba_core._select_indices(data, deviations, [], 2, 0, workers=1)

    assert numba.get_num_threads() == previous_workers


def test_public_api_selects_with_numba():
    columns = (
        DataColumn([0, 0, 4, 4]),
        DataColumn([350, 10, 180, 90], directional=True),
    )

    expected_data, expected_indices = max_diss_alg(*columns, n_clusters=3)
    actual_data, actual_indices = max_diss_alg(
        *columns,
        n_clusters=3,
        engine="numba",
    )

    np.testing.assert_array_equal(actual_data, expected_data)
    assert actual_indices == expected_indices == [0, 2, 3]


@pytest.mark.parametrize("workers", [True, 1.5, "2"])
def test_numba_rejects_non_integer_workers(workers):
    data = np.asarray([[0.0], [1.0]])

    with pytest.raises(TypeError, match="workers must be an integer or None"):
        numba_core._select_indices(data, np.asarray([0.5]), [], 1, 0, workers=workers)


@pytest.mark.parametrize("workers", [-1, 0])
def test_numba_rejects_non_positive_workers(workers):
    data = np.asarray([[0.0], [1.0]])

    with pytest.raises(ValueError, match="workers must be a positive integer"):
        numba_core._select_indices(data, np.asarray([0.5]), [], 1, 0, workers=workers)
