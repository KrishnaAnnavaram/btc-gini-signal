"""Forecast models. Each learned model is a scikit-learn Pipeline, so imputation and scaling fit on training rows only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RandomWalk(BaseEstimator, RegressorMixin):
    """Predicts a zero log return: tomorrow's price equals today's price."""

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.zeros(len(X))


class HistoricalMean(BaseEstimator, RegressorMixin):
    """Predicts the mean training return (drift)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _pipe(model) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    factory: Callable[[int], object]
    uses_features: bool = True
    feature_override: tuple[str, ...] | None = None


MODELS: dict[str, ModelSpec] = {
    "random_walk": ModelSpec("random_walk", "Zero return (the naive baseline)", lambda s: RandomWalk(), False),
    "mean": ModelSpec("mean", "Historical mean return", lambda s: HistoricalMean(), False),
    "ar": ModelSpec("ar", "AR(3): OLS on the last three daily returns", lambda s: _pipe(LinearRegression()),
                    True, ("ret_lag1", "ret_lag2", "ret_lag3")),
    "ridge": ModelSpec("ridge", "Ridge regression, alpha chosen by internal CV",
                       lambda s: _pipe(RidgeCV(alphas=np.logspace(-2, 4, 13)))),
    "gbm": ModelSpec("gbm", "Histogram gradient boosting (shallow, strong regularisation)",
                     lambda s: _pipe(HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=150,
                                                                   min_samples_leaf=40, l2_regularization=1.0,
                                                                   random_state=s))),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]


def columns_for(spec: ModelSpec, feature_cols: list[str]) -> list[str]:
    if spec.feature_override is not None:
        return list(spec.feature_override)
    return list(feature_cols)
