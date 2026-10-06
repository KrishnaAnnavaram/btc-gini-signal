"""Synthetic daily OHLCV with volatility clustering, jumps and an optional planted inequality signal.

With `signal=0` the expected next-day return is zero, so no feature can beat the random walk
except by chance. With `signal>0` the next-day drift depends on the Gini index of recent
absolute returns. Tests use this to check that the harness finds a real effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _gini_sorted(x: np.ndarray) -> float:
    total = x.sum()
    if total <= 0:
        return 0.0
    n = x.size
    ranks = np.arange(1, n + 1)
    return float(2 * np.sum(ranks * x) / (n * total) - (n + 1) / n)


def generate(n_days: int = 2500, seed: int = 42, start: str = "2017-01-01", signal: float = 0.0,
             window: int = 30, start_price: float = 1000.0) -> pd.DataFrame:
    if n_days < window + 10:
        raise ValueError(f"n_days must be >= {window + 10}")
    rng = np.random.default_rng(seed)
    long_run_var = 0.035 ** 2
    alpha, beta = 0.08, 0.90
    omega = long_run_var * (1 - alpha - beta)
    var = long_run_var
    r = np.zeros(n_days)
    for t in range(1, n_days):
        var = omega + alpha * r[t - 1] ** 2 + beta * var
        sigma = np.sqrt(var)
        mu = 0.0
        if signal and t > window:
            g = _gini_sorted(np.sort(np.abs(r[t - window:t])))
            mu = float(np.clip(signal * (g - 0.45), -1.0, 1.0)) * sigma
        jump = rng.normal(0.0, 3 * sigma) if rng.random() < 0.03 else 0.0
        r[t] = mu + sigma * rng.standard_normal() + jump

    close = start_price * np.exp(np.cumsum(r))
    prev = np.concatenate([[start_price], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.002, n_days))
    spread = np.abs(rng.normal(0, 0.4, n_days)) * np.sqrt(long_run_var)
    high = np.maximum(open_, close) * np.exp(spread)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.4, n_days)) * np.sqrt(long_run_var))
    volume = np.exp(rng.normal(10, 0.5, n_days))
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
