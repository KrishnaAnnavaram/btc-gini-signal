"""Lorenz curve, Gini index and Kolkata index, for one sample and on a rolling window.

Definitions for non-negative values x_1..x_n:

* Lorenz curve L(p): the share of the total held by the smallest fraction p of the values.
* Gini index G = 1 - 2 * (area under L). 0 = all values equal, (n-1)/n = one value holds all.
* Kolkata index k: the x-coordinate where L(p) = 1 - p. The largest (1 - k) fraction of
  values holds the fraction k of the total. k = 0.5 means equality, k -> 1 means extreme
  inequality (Ghosh, Chattopadhyay and Chakrabarti, 2014).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _check(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("need a non-empty 1-D sample")
    if np.isnan(arr).any():
        raise ValueError("sample contains NaN")
    if (arr < 0).any():
        raise ValueError("inequality indices need non-negative values (use absolute returns)")
    return arr


def lorenz_curve(x) -> tuple[np.ndarray, np.ndarray]:
    """Points (p, L) of the Lorenz curve, starting at (0, 0) and ending at (1, 1)."""
    arr = np.sort(_check(x))
    n = arr.size
    total = arr.sum()
    p = np.arange(n + 1) / n
    if total == 0:
        return p, p.copy()
    share = np.concatenate([[0.0], np.cumsum(arr) / total])
    return p, share


def gini(x) -> float:
    arr = np.sort(_check(x))
    total = arr.sum()
    if total == 0:
        return 0.0
    n = arr.size
    ranks = np.arange(1, n + 1)
    return float(2 * np.sum(ranks * arr) / (n * total) - (n + 1) / n)


def kolkata(x) -> float:
    p, L = lorenz_curve(x)
    f = L - (1 - p)  # increases from -1 at p=0 to +1 at p=1
    i = int(np.argmax(f >= 0))
    if i == 0:
        return 0.0
    # linear interpolation between point i-1 (f<0) and point i (f>=0)
    t = -f[i - 1] / (f[i] - f[i - 1])
    return float(p[i - 1] + t * (p[i] - p[i - 1]))


def rolling_gini_kolkata(values, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Gini and Kolkata of values[t-window+1 .. t] for each t. The first window-1 entries are NaN.

    The value at t uses only entries up to t, so it never sees the future.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    g = np.full(n, np.nan)
    k = np.full(n, np.nan)
    if n < window:
        return g, k
    win = np.sort(sliding_window_view(arr, window), axis=1)
    valid = ~np.isnan(win).any(axis=1)
    win = np.where(np.isnan(win), 0.0, win)
    total = win.sum(axis=1)
    ranks = np.arange(1, window + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gv = 2 * (win * ranks).sum(axis=1) / (window * total) - (window + 1) / window
        cum = np.cumsum(win, axis=1) / total[:, None]
    gv = np.where(total > 0, gv, 0.0)
    share = np.concatenate([np.zeros((cum.shape[0], 1)), cum], axis=1)
    p = np.arange(window + 1) / window
    f = share - (1 - p)
    i = np.argmax(f >= 0, axis=1)
    i = np.clip(i, 1, window)
    rows = np.arange(f.shape[0])
    f0, f1 = f[rows, i - 1], f[rows, i]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(f1 != f0, -f0 / (f1 - f0), 0.0)
    kv = p[i - 1] + t * (p[i] - p[i - 1])
    kv = np.where(total > 0, kv, 0.5)
    g[window - 1:] = np.where(valid, gv, np.nan)
    k[window - 1:] = np.where(valid, kv, np.nan)
    return g, k
