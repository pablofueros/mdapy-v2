import numpy as np
import pytest

from mdapy import DataColumn, max_diss_alg

pytest.importorskip("numba")


@pytest.mark.parametrize(
    ("columns", "n_clusters", "seed_index", "expected_indices"),
    [
        ((DataColumn([-2, 0, 2]),), 3, 1, [1, 0, 2]),
        ((DataColumn([0, 0, 2, 2]),), 4, 0, [0, 2, 1, 3]),
        (
            (
                DataColumn([0, 0, 4, 4, 0]),
                DataColumn([350, 10, 180, 90, 350], directional=True),
            ),
            5,
            0,
            [0, 2, 3, 1, 4],
        ),
        (
            (
                DataColumn([0, 1, 2, 3]),
                DataColumn([-370, 350, 730, 180], directional=True),
            ),
            4,
            0,
            [0, 3, 2, 1],
        ),
        ((DataColumn([0, 2, 5, 9]),), 3, 2, [2, 0, 3]),
        ((DataColumn([0, 1]),), 1, 1, [1]),
    ],
    ids=[
        "ties",
        "duplicates",
        "directional",
        "wrapped-angles",
        "seed",
        "single-selection",
    ],
)
def test_numba_matches_numpy_reference_cases(
    columns, n_clusters, seed_index, expected_indices
):
    numpy_points, numpy_indices = max_diss_alg(
        *columns,
        n_clusters=n_clusters,
        seed_index=seed_index,
        engine="numpy",
    )
    numba_points, numba_indices = max_diss_alg(
        *columns,
        n_clusters=n_clusters,
        seed_index=seed_index,
        engine="numba",
    )

    assert numba_indices == numpy_indices == expected_indices
    np.testing.assert_array_equal(numba_points, numpy_points)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numba_preserves_nearby_distance_order(dtype):
    columns = (
        DataColumn(np.asarray([0, 1, 1, 0, 0, 0, 0, 0], dtype=dtype)),
        DataColumn(np.asarray([0, 0, 0.000173, -0.5, 0.5, -0.5, 0.5, 0], dtype=dtype)),
    )

    numpy_points, numpy_indices = max_diss_alg(
        *columns,
        n_clusters=3,
        engine="numpy",
    )
    numba_points, numba_indices = max_diss_alg(
        *columns,
        n_clusters=3,
        engine="numba",
    )

    assert numba_indices == numpy_indices == [0, 2, 3]
    np.testing.assert_array_equal(numba_points, numpy_points)


@pytest.mark.parametrize(
    ("values", "seed_index", "expected_indices"),
    [
        (np.array([-128, 0, 127], dtype=np.int8), 0, [0, 2, 1]),
        (np.array([0, 1, 255], dtype=np.uint8), 2, [2, 0, 1]),
        (np.array([0, 1, 3], dtype=np.float16), 0, [0, 2, 1]),
        (np.array([0, 1, 3], dtype=np.longdouble), 0, [0, 2, 1]),
    ],
    ids=["signed-overflow", "unsigned-wraparound", "float16", "extended-float"],
)
def test_numba_uses_float64_calculations_for_other_dtypes(
    values, seed_index, expected_indices
):
    numpy_points, numpy_indices = max_diss_alg(
        DataColumn(values),
        n_clusters=3,
        seed_index=seed_index,
        engine="numpy",
    )
    numba_points, numba_indices = max_diss_alg(
        DataColumn(values),
        n_clusters=3,
        seed_index=seed_index,
        engine="numba",
    )

    assert numba_indices == numpy_indices == expected_indices
    assert numba_points.dtype == numpy_points.dtype == values.dtype
    np.testing.assert_array_equal(numba_points, numpy_points)
