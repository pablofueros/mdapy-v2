import numpy as np
import pytest

from mdapy import DataColumn, max_diss_alg


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("directional", [False, True])
def test_column_owns_data_and_keeps_cached_deviation_consistent(dtype, directional):
    original = [350, 10, 180, 90] if directional else [0, 2, 5, 9]
    expected_indices = [0, 2, 3] if directional else [0, 3, 2]
    values = np.array(original, dtype=dtype)
    column = DataColumn(values, directional=directional)
    deviation = column.standard_deviation

    assert column.values.dtype == values.dtype
    assert not np.shares_memory(column.values, values)
    assert values.flags.writeable
    values[:] = 0

    np.testing.assert_array_equal(column.values, original)
    assert column.standard_deviation == deviation
    points, indices = max_diss_alg(column, n_clusters=3)
    assert indices == expected_indices
    assert points.dtype == values.dtype
    np.testing.assert_array_equal(points[:, 0], np.array(original)[expected_indices])


def test_column_rejects_writes_through_public_array():
    column = DataColumn([0, 2, 5, 9])
    deviation = column.standard_deviation
    values = column.values

    with pytest.raises(ValueError, match="read-only"):
        values[0] = 100
    with pytest.raises(ValueError, match="WRITEABLE"):
        values.setflags(write=True)

    np.testing.assert_array_equal(column.values, [0, 2, 5, 9])
    assert column.standard_deviation == deviation


@pytest.mark.parametrize(
    "attribute, replacement", [("values", np.array([1, 2, 3])), ("directional", True)]
)
def test_column_public_properties_cannot_be_reassigned(attribute, replacement):
    column = DataColumn([0, 1, 2])
    with pytest.raises(AttributeError, match=attribute):
        setattr(column, attribute, replacement)


def test_column_api_selects_expected_points_without_progress(capsys):
    values = [0, 2, 5, 9]
    selected, indices = max_diss_alg(DataColumn(values), n_clusters=3)

    assert indices == [0, 3, 2]
    np.testing.assert_array_equal(selected[:, 0], [0, 9, 5])
    assert capsys.readouterr().out == ""


def test_column_api_can_show_progress(capsys):
    max_diss_alg(DataColumn([0, 1, 2]), n_clusters=2, show_progress=True)

    assert "Finding cluster nº2" in capsys.readouterr().out


@pytest.mark.parametrize("show_progress", [1, None])
def test_column_api_rejects_non_boolean_progress(show_progress):
    with pytest.raises(TypeError, match="show_progress must be a bool"):
        max_diss_alg(
            DataColumn([0, 1, 2]),
            n_clusters=2,
            show_progress=show_progress,
        )


def test_numpy_is_the_default_engine():
    columns = (DataColumn([0, 2, 5, 9]),)

    default_result = max_diss_alg(*columns, n_clusters=3, seed_index=2)
    numpy_result = max_diss_alg(
        *columns,
        n_clusters=3,
        seed_index=2,
        engine="numpy",
    )

    np.testing.assert_array_equal(default_result[0], numpy_result[0])
    assert default_result[1] == numpy_result[1]


@pytest.mark.parametrize("engine", ["python", "rust", "NUMPY", True, None])
def test_column_api_rejects_unknown_engine(engine):
    with pytest.raises(ValueError, match="engine must be either 'numpy' or 'numba'"):
        max_diss_alg(DataColumn([0, 1]), n_clusters=1, engine=engine)


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        (
            (([0, 1, 2],), 2),
            {},
            "required keyword-only argument: 'n_clusters'",
        ),
        (
            (([0, 1, 2],),),
            {"n_clusters": 2},
            "from mdapy.legacy import max_diss_alg",
        ),
        (
            (),
            {"data_tuple": ([0, 1, 2],), "n_clusters": 2},
            "from mdapy.legacy import max_diss_alg",
        ),
        (
            (([0, 1, 2],),),
            {"n_clusters": 2, "dir_indices": [0]},
            "from mdapy.legacy import max_diss_alg",
        ),
    ],
)
def test_old_api_call_points_to_legacy(args, kwargs, message):
    with pytest.raises(TypeError, match=message):
        max_diss_alg(*args, **kwargs)


def test_column_api_rejects_out_of_range_cluster_count():
    column = DataColumn([0, 1, 2])
    with pytest.raises(ValueError, match="n_clusters must be between"):
        max_diss_alg(column, n_clusters=4)


def test_column_api_rejects_invalid_columns():
    with pytest.raises(ValueError, match="same length"):
        max_diss_alg(DataColumn([0, 1]), DataColumn([0, 1, 2]), n_clusters=1)
    with pytest.raises(ValueError, match="finite"):
        DataColumn([0, np.inf])
    empty = DataColumn([])
    with pytest.raises(ValueError, match="n_clusters must be between"):
        max_diss_alg(empty, n_clusters=1)


@pytest.mark.parametrize("directional", [False, True])
def test_api_rejects_zero_deviation(directional):
    column = DataColumn([1, 1], directional=directional)
    with pytest.raises(ValueError, match="zero or undefined deviation"):
        max_diss_alg(column, n_clusters=1)


def test_columns_use_numpy_common_dtype():
    integers = DataColumn(np.array([0, 1, 2], dtype=np.int32))
    decimals = DataColumn(np.array([0, 1, 2], dtype=np.float32))

    selected, _ = max_diss_alg(integers, decimals, n_clusters=2)

    assert selected.dtype == np.result_type(integers.values, decimals.values)
