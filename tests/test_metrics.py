import pytest
from metrics import average_precision, mean_average_precision


@pytest.mark.parametrize("y_true, y_pred, k, expected", [
    # number of data < k
    ([1, 1, 1, 0, 0, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 10, 1.0),
    ([1, 0, 1, 0, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 10, (1 + 2 / 3 + 3 / 5) / 3),
    ([0, 1, 0, 1, 0, 1], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 10, (1 / 2 + 2 / 4 + 3 / 6) / 3),
    # number of pos < k < number of data
    ([1, 1, 1, 0, 0, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 4, 1.0),
    ([1, 0, 1, 0, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 4, (1 + 2 / 3) / 3),
    ([0, 1, 0, 1, 0, 1], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 4, (1 / 2 + 2 / 4) / 3),
    # k < number of pos
    ([1, 1, 1, 0, 0, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 2, 1.0),
    ([1, 0, 1, 0, 1, 0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 2, 1 / 2),
    ([0, 1, 0, 1, 0, 1], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 2, 1 / 2 / 2),

])
def test_average_precision(y_true, y_pred, k, expected):
    result = average_precision(y_true, y_pred, k)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("y_true, y_pred, k, n_jobs, expected", [
    (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
        [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        2, 1, (1.0 + 0.5 + 0.0) / 3.0,
    ),
    (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
        [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        2, -1, (1.0 + 0.5 + 0.0) / 3.0,
    )
])
def test_mean_average_precision(y_true, y_pred, k, n_jobs, expected):
    result = mean_average_precision(y_true, y_pred, k, n_jobs)
    assert result == pytest.approx(expected)
