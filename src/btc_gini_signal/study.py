"""The ablation study: each model with and without the inequality features, per horizon, walk-forward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics
from .config import Settings
from .data import PriceData
from .features import FEATURE_SETS, build_features, make_target
from .models import get_model
from .walkforward import WalkForwardResult, walk_forward

DEFAULT_MODELS = ("random_walk", "mean", "ar", "ridge", "gbm")


@dataclass
class StudyResult:
    rows: list[dict]
    ablation: list[dict]
    backtests: list[dict]
    results: dict[tuple[int, str, str], WalkForwardResult]


def _aligned(a: WalkForwardResult, b: WalkForwardResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = a.dates.intersection(b.dates)
    ia = a.dates.get_indexer(common)
    ib = b.dates.get_indexer(common)
    return a.y_true[ia], a.y_pred[ia], b.y_pred[ib]


def run_study(data: PriceData, settings: Settings, models=DEFAULT_MODELS,
              feature_sets=("base", "base+inequality")) -> StudyResult:
    features = build_features(data.frame, settings.window)
    # one common evaluation window: rows where every feature (of every set) is complete
    rows, ablation, backtests = [], [], []
    results: dict[tuple[int, str, str], WalkForwardResult] = {}
    all_cols = sorted({c for fs in feature_sets for c in FEATURE_SETS[fs]})
    for h in settings.horizons:
        target = make_target(data.frame, h)
        for model in models:
            spec = get_model(model)
            sets = feature_sets if spec.uses_features and spec.feature_override is None else (feature_sets[0],)
            for fs in sets:
                # rows with any missing feature of the union are skipped by every run, so all runs share rows
                masked = features.copy()
                masked.loc[masked[all_cols].isna().any(axis=1), FEATURE_SETS[fs]] = np.nan
                res = walk_forward(masked, target, FEATURE_SETS[fs], model, h, settings.min_train,
                                   settings.refit_every, settings.seed, feature_set=fs)
                results[(h, model, fs)] = res
                summary = metrics.summarize(res.y_true, res.y_pred, horizon=h)
                rows.append({"horizon": h, "model": model, "feature_set": fs if spec.uses_features else "-",
                             "refits": len(res.refits), **summary})
                if h == 1 and model != "random_walk":
                    bt = metrics.backtest_sign(res.y_true, res.y_pred, settings.cost_bps)
                    backtests.append({"model": model, "feature_set": fs if spec.uses_features else "-",
                                      **bt.to_dict()})
        for model in models:
            a, b = results.get((h, model, "base")), results.get((h, model, "base+inequality"))
            if a is None or b is None:
                continue
            y, pa, pb = _aligned(a, b)
            dm = metrics.diebold_mariano(y, pb, pa, h)
            ablation.append({"horizon": h, "model": model, "n": int(y.size),
                             "rmse_base": metrics.rmse(y, pa), "rmse_with_inequality": metrics.rmse(y, pb),
                             "dm_statistic": dm.statistic, "p_value": dm.p_value})
    return StudyResult(rows, ablation, backtests, results)
