"""A real multi-step forecast with the direct method: one model per horizon, fit on all observed targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from .data import PriceData
from .features import FEATURE_SETS, build_features, make_target
from .models import columns_for, get_model


def direct_forecast(data: PriceData, horizons: list[int], model: str = "ridge", feature_set: str = "base+inequality",
                    window: int = 30, seed: int = 0) -> pd.DataFrame:
    """Forecast log returns and price levels for the last observed day + h, for each h.

    For horizon h, the model trains on the rows whose h-day target is already observed,
    then predicts from the features of the last day. No forecast feeds another forecast.
    """
    spec = get_model(model)
    features = build_features(data.frame, window)
    cols = columns_for(spec, FEATURE_SETS[feature_set])
    X = features[cols]
    last_date = features.index[-1]
    last_close = float(data.frame["close"].iloc[-1])
    if X.iloc[[-1]].isna().any(axis=1).iloc[0]:
        raise ValueError("the last day has incomplete features; the series is too short")
    rows = []
    for h in sorted(set(horizons)):
        y = make_target(data.frame, h).to_numpy()
        train = (~X.isna().any(axis=1).to_numpy()) & ~np.isnan(y)
        if train.sum() < 50:
            raise ValueError(f"too few rows to train horizon {h}")
        est = clone(spec.factory(seed)).fit(X[train], y[train])
        r = float(est.predict(X.iloc[[-1]])[0])
        rows.append({"origin_date": pd.Timestamp(last_date).date().isoformat(), "horizon_days": h,
                     "target_date": (pd.Timestamp(last_date) + pd.Timedelta(days=h)).date().isoformat(),
                     "predicted_log_return": r, "predicted_close": last_close * float(np.exp(r)),
                     "last_close": last_close, "train_rows": int(train.sum())})
    return pd.DataFrame(rows)
