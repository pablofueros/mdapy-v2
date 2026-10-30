import builtins
import sys
from typing import Any

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


def test_column_api_selects_expected_points_and_is_silent(capsys):
    values = [0, 2, 5, 9]
    selected, indices = max_diss_alg(DataColumn(values), n_clusters=3)

    assert indices == [0, 3, 2]
    np.testing.assert_array_equal(selected[:, 0], [0, 9, 5])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("show_progress", [False, True])
def test_column_api_accepts_and_ignores_progress_option(show_progress, capsys):
    selected, indices = max_diss_alg(
        DataColumn([0, 1, 2]),
        n_clusters=2,
        show_progress=show_progress,
    )

    assert indices == [0, 2]
    np.testing.assert_array_equal(selected[:, 0], [0, 2])
    assert capsys.readouterr().out == ""


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


@pytest.mark.parametrize(
    "values",
    [
        np.array([False, True]),
        np.array([0, 1], dtype=np.complex128),
        np.array(["0", "1"]),
        np.array([0, 1], dtype=object),
        np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        np.array([0, 1], dtype="timedelta64[D]"),
    ],
    ids=["bool", "complex", "string", "object", "datetime", "timedelta"],
)
def test_column_rejects_non_real_numeric_dtypes(values):
    with pytest.raises(TypeError, match="real numeric dtype"):
        DataColumn(values)


def test_column_rejects_non_boolean_directional_flag():
    directional: Any = 1
    with pytest.raises(TypeError, match="directional must be a bool"):
        DataColumn([0, 1], directional=directional)


def test_column_rejects_non_one_dimensional_values():
    with pytest.raises(ValueError, match="one-dimensional"):
        DataColumn([[0, 1], [2, 3]])


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


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    ],
)
def test_real_numeric_dtypes_preserve_stored_values(dtype):
    values = np.array([0, 1, 3], dtype=dtype)

    selected, indices = max_diss_alg(DataColumn(values), n_clusters=3)

    assert indices == [0, 2, 1]
    assert selected.dtype == values.dtype
    np.testing.assert_array_equal(selected[:, 0], values[indices])


def test_column_api_requires_at_least_one_column():
    with pytest.raises(ValueError, match="At least one DataColumn"):
        max_diss_alg(n_clusters=1)


@pytest.mark.parametrize("n_clusters", [True, 1.5, "1"])
def test_column_api_rejects_non_integer_cluster_count(n_clusters):
    with pytest.raises(TypeError, match="n_clusters must be an integer"):
        max_diss_alg(DataColumn([0, 1]), n_clusters=n_clusters)


@pytest.mark.parametrize("seed_index", [True, 1.5, "1"])
def test_column_api_rejects_non_integer_seed_index(seed_index):
    with pytest.raises(TypeError, match="seed_index must be an integer"):
        max_diss_alg(DataColumn([0, 1]), n_clusters=1, seed_index=seed_index)


@pytest.mark.parametrize("seed_index", [-1, 2])
def test_column_api_rejects_out_of_bounds_seed_index(seed_index):
    with pytest.raises(ValueError, match="seed_index is out of bounds"):
        max_diss_alg(DataColumn([0, 1]), n_clusters=1, seed_index=seed_index)


def test_column_api_rejects_unexpected_keyword_argument():
    with pytest.raises(TypeError, match="Unexpected keyword argument: workers"):
        max_diss_alg(DataColumn([0, 1]), n_clusters=1, workers=1)


def test_numba_engine_explains_how_to_install_optional_dependency(monkeypatch):
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "mdapy.numba", raising=False)

    def import_without_numba(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numba":
            raise ModuleNotFoundError("No module named 'numba'", name="numba")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_numba)

    with pytest.raises(ImportError, match=r'pip install "mdapy-v2\[numba\]"'):
        max_diss_alg(DataColumn([0, 1]), n_clusters=1, engine="numba")


def test_numba_engine_preserves_unexpected_import_errors(monkeypatch):
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "mdapy.numba", raising=False)

    def import_with_broken_numba_dependency(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "numba":
            raise ModuleNotFoundError("No module named 'llvmlite'", name="llvmlite")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_numba_dependency)

    with pytest.raises(ModuleNotFoundError, match="llvmlite"):
        max_diss_alg(DataColumn([0, 1]), n_clusters=1, engine="numba")
