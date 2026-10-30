import numpy as np
import pytest

from mdapy.legacy import max_diss_alg


def test_legacy_api_adapts_tuple_seed_and_directional_indices(capsys):
    heights = [1.0, 2.0, 5.0, 3.0]
    directions = [350.0, 10.0, 180.0, 90.0]

    selected, indices = max_diss_alg(
        (heights, directions),
        n_clusters=3,
        seed_index=1,
        dir_indices=[1],
    )

    assert indices == [1, 2, 3]
    assert selected.dtype == np.float64
    np.testing.assert_array_equal(selected, [[2, 10], [5, 180], [3, 90]])
    assert capsys.readouterr().out == ""


def test_legacy_api_accepts_and_ignores_progress_option(capsys):
    selected, indices = max_diss_alg(
        ([0, 1, 2],),
        n_clusters=2,
        show_progress=True,
    )

    assert indices == [0, 2]
    np.testing.assert_array_equal(selected[:, 0], [0, 2])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("dir_indices", [[True], [1.0]])
def test_legacy_rejects_non_integer_directional_indices(dir_indices):
    with pytest.raises(TypeError, match="dir_indices must contain only integers"):
        max_diss_alg(([0, 1],), n_clusters=1, dir_indices=dir_indices)


def test_legacy_rejects_out_of_range_directional_index():
    with pytest.raises(ValueError, match="out-of-bounds"):
        max_diss_alg(([0, 1],), n_clusters=1, dir_indices=[1])
