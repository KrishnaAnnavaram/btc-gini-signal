"""Walk-forward evaluation with an expanding window and a target embargo.

At a refit day s, the model trains on rows t with t + h <= s. Only those targets are already
observed on day s. The model then predicts rows s .. s + refit_every - 1. No row is used to
train a model that predicts it or any earlier row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from .models import columns_for, get_model


@dataclass
class Refit:
    origin: int          # position of the first predicted row
    train_last: int      # position of the last training row (origin - horizon)
    n_train: int


@dataclass
class WalkForwardResult:
    model: str
    feature_set: str
    horizon: int
    dates: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    refits: list[Refit] = field(default_factory=list)


def walk_forward(features: pd.DataFrame, target: pd.Series, feature_cols: list[str], model: str, horizon: int,
                 min_train: int = 730, refit_every: int = 30, seed: int = 0,
                 feature_set: str = "") -> WalkForwardResult:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    spec = get_model(model)
    cols = columns_for(spec, feature_cols)
    X_all = features[cols]
    y_all = target.to_numpy(dtype=float)
    # rows with complete features (after the warm-up) and an observed target
    usable = (~X_all.isna().any(axis=1).to_numpy()) & ~np.isnan(y_all)
    positions = np.flatnonzero(usable)
    if positions.size <= min_train:
        raise ValueError(f"only {positions.size} usable rows, need more than min_train={min_train}")
    first_origin = positions[min_train]
    test_positions = positions[positions >= first_origin]

    preds = np.full(len(y_all), np.nan)
    refits: list[Refit] = []
    i = 0
    while i < test_positions.size:
        origin = test_positions[i]
        block = test_positions[(test_positions >= origin) & (test_positions < origin + refit_every)]
        train_mask = usable.copy()
        train_mask[origin - horizon + 1:] = False  # embargo: train rows satisfy t + h <= origin
        train_idx = np.flatnonzero(train_mask)
        est = clone(spec.factory(seed)).fit(X_all.iloc[train_idx], y_all[train_idx])
        preds[block] = est.predict(X_all.iloc[block])
        refits.append(Refit(int(origin), int(train_idx[-1]), int(train_idx.size)))
        i += block.size

    keep = ~np.isnan(preds)
    return WalkForwardResult(model, feature_set, horizon, features.index[keep], y_all[keep], preds[keep], refits)
