"""Feature table and forecast targets. Every feature at day t uses only prices up to day t.

Targets are future log returns y_h(t) = log(close[t+h] / close[t]) for h >= 1. Features and
targets share one date index, so a model never trains on one target and scores on another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inequality import rolling_gini_kolkata

BASE_FEATURES = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_7d", "vol_7", "vol_30",
                 "skew_30", "kurt_30", "range_log"]
INEQUALITY_FEATURES = ["gini_abs_ret", "kolkata_abs_ret", "gini_change_7"]
FEATURE_SETS = {
    "base": BASE_FEATURES,
    "base+inequality": BASE_FEATURES + INEQUALITY_FEATURES,
    "inequality": INEQUALITY_FEATURES,
}


def build_features(prices: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Return a frame indexed by date with BASE_FEATURES + INEQUALITY_FEATURES (NaN in warm-up rows)."""
    close = prices["close"].to_numpy(dtype=float)
    r = pd.Series(np.log(close), index=prices["date"]).diff()
    out = pd.DataFrame(index=prices["date"].to_numpy())
    out.index.name = "date"
    out["ret_lag1"] = r.to_numpy()
    out["ret_lag2"] = r.shift(1).to_numpy()
    out["ret_lag3"] = r.shift(2).to_numpy()
    out["ret_lag5"] = r.shift(4).to_numpy()
    out["ret_7d"] = r.rolling(7).sum().to_numpy()
    out["vol_7"] = r.rolling(7).std().to_numpy()
    out["vol_30"] = r.rolling(30).std().to_numpy()
    out["skew_30"] = r.rolling(30).skew().to_numpy()
    out["kurt_30"] = r.rolling(30).kurt().to_numpy()
    out["range_log"] = np.log(prices["high"].to_numpy(dtype=float) / prices["low"].to_numpy(dtype=float))
    g, k = rolling_gini_kolkata(np.abs(r.to_numpy()), window)
    out["gini_abs_ret"] = g
    out["kolkata_abs_ret"] = k
    out["gini_change_7"] = out["gini_abs_ret"] - out["gini_abs_ret"].shift(7)
    return out


def make_target(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """y_h(t) = log(close[t+h] / close[t]). The last `horizon` rows are NaN (not yet observed)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1 day")
    logc = np.log(prices["close"].to_numpy(dtype=float))
    y = np.full(logc.size, np.nan)
    y[:-horizon] = logc[horizon:] - logc[:-horizon]
    return pd.Series(y, index=prices["date"].to_numpy(), name=f"y_{horizon}d")
