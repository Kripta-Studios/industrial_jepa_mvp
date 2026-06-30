import numpy as np

from sensor_jepa.data.windowing import sliding_windows, stratified_label_fraction_indices


def test_sliding_windows_shape_and_indices():
    x = np.arange(20).reshape(10, 2)
    windows, idx = sliding_windows(x, window_length=4, stride=2)
    assert windows.shape == (4, 4, 2)
    assert idx.tolist() == [3, 5, 7, 9]
    np.testing.assert_array_equal(windows[0], x[:4])


def test_label_fraction_keeps_each_class():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    idx = stratified_label_fraction_indices(y, 0.1, seed=1)
    assert set(y[idx]) == {0, 1, 2}

