from functools import partial
from typing import Callable

import numpy as np
from joblib import Parallel, delayed


def average_precision(y_true: list[int], y_pred: list[float], k: int) -> float:
    k = min(k, len(y_true))
    num_pos = np.sum(y_true)
    if num_pos == 0:
        return np.nan

    ordered_idxs = np.argsort(y_pred)[::-1][:k]
    y = np.array(y_true)[ordered_idxs]
    y_cumsum = np.cumsum(y)
    y_denom = np.arange(1, k + 1)
    return np.sum(y * y_cumsum / y_denom) / min(num_pos, k)


def mean_score(y_true: list[list], y_pred: list[list], metric: Callable[[list, list], float], n_jobs: int = 1) -> float:
    if n_jobs == 1:
        return np.nanmean([metric(a, b) for a, b in zip(y_true, y_pred)])
    else:
        result = Parallel(n_jobs=n_jobs)([delayed(metric)(a, b) for a, b in zip(y_true, y_pred)])
        return np.nanmean(result)


def mean_average_precision(y_true: list[list[int]], y_pred: list[list[float]], k: int, n_jobs: int = 1) -> float:
    return mean_score(y_true, y_pred, partial(average_precision, k=k), n_jobs)
