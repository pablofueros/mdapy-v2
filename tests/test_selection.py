import numpy as np
import pytest

from mdapy import DataColumn, max_diss_alg


def test_directional_column_uses_shortest_angular_distance():
    values = [350, 10, 180]

    _, indices = max_diss_alg(DataColumn(values, directional=True), n_clusters=2)

    assert indices == [0, 2]


def test_mixed_columns_select_expected_points():
    heights = [1.0, 2.0, 5.0, 3.0]
    directions = [350.0, 10.0, 180.0, 90.0]

    selected, indices = max_diss_alg(
        DataColumn(heights),
        DataColumn(directions, directional=True),
        n_clusters=3,
        seed_index=1,
    )

    assert indices == [1, 2, 3]
    np.testing.assert_array_equal(selected, [[2, 10], [5, 180], [3, 90]])


def test_seed_index_changes_selection_order():
    selected, indices = max_diss_alg(
        DataColumn([0, 2, 5, 9]), n_clusters=3, seed_index=2
    )

    assert indices == [2, 0, 3]
    assert selected.shape == (3, 1)
    np.testing.assert_array_equal(selected[:, 0], [5, 0, 9])


def test_equal_distances_select_first_input_index():
    _, indices = max_diss_alg(DataColumn([-2, 0, 2]), n_clusters=3, seed_index=1)

    assert indices == [1, 0, 2]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nearby_distances_do_not_become_an_artificial_tie(dtype):
    _, indices = max_diss_alg(
        DataColumn(np.asarray([0, 1, 1, 0, 0, 0, 0, 0], dtype=dtype)),
        DataColumn(np.asarray([0, 0, 0.000173, -0.5, 0.5, -0.5, 0.5, 0], dtype=dtype)),
        n_clusters=3,
    )

    assert indices == [0, 2, 3]


def test_mixed_columns_select_by_nearest_selected_point():
    # From 350°, 10° is only 20° away. After selecting 180°, the point at
    # 90° is farther from its nearest selected point than the point at 10°.
    selected, indices = max_diss_alg(
        DataColumn([0, 0, 4, 4]),
        DataColumn([350, 10, 180, 90], directional=True),
        n_clusters=3,
    )

    assert indices == [0, 2, 3]
    np.testing.assert_array_equal(selected, [[0, 350], [4, 180], [4, 90]])


def test_rescaling_one_column_preserves_selection_order():
    linear = [0, 3, 1, 2]
    angles = DataColumn([0, 10, 20, 170], directional=True)

    _, original_indices = max_diss_alg(DataColumn(linear), angles, n_clusters=3)
    _, scaled_indices = max_diss_alg(
        DataColumn([100 * value for value in linear]), angles, n_clusters=3
    )

    assert original_indices == scaled_indices == [0, 3, 1]


def test_duplicate_observations_keep_input_order():
    _, indices = max_diss_alg(DataColumn([0, 0, 2, 2]), n_clusters=4)

    assert indices == [0, 2, 1, 3]


def test_multiple_directional_columns():
    _, indices = max_diss_alg(
        DataColumn([350, 10, 180, 90, 350], directional=True),
        DataColumn([5, 355, 180, 270, 5], directional=True),
        n_clusters=5,
    )

    assert indices == [0, 2, 3, 1, 4]


def test_directional_values_can_wrap_multiple_turns():
    _, indices = max_diss_alg(
        DataColumn([0, 1, 2, 3]),
        DataColumn([-370, 350, 730, 180], directional=True),
        n_clusters=4,
    )

    assert indices == [0, 3, 2, 1]


@pytest.mark.parametrize("scale", [1e-30, 1e20])
def test_algorithm_preserves_float64_values(scale):
    values = np.array([0, scale, 2 * scale], dtype=np.float64)

    selected, indices = max_diss_alg(DataColumn(values), n_clusters=3)

    assert indices == [0, 2, 1]
    assert selected.dtype == np.float64
    np.testing.assert_array_equal(selected[:, 0], values[indices])


def test_float64_keeps_distinct_values_beyond_float32_precision():
    values = np.array([100_000_000, 100_000_001, 100_000_002], dtype=np.float64)

    selected, indices = max_diss_alg(DataColumn(values), n_clusters=3)

    assert indices == [0, 2, 1]
    np.testing.assert_array_equal(selected[:, 0], values[indices])
