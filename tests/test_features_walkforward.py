import numpy as np
import pandas as pd
import pytest

from btc_gini_signal import synthetic
from btc_gini_signal.features import BASE_FEATURES, FEATURE_SETS, INEQUALITY_FEATURES, build_features, make_target
from btc_gini_signal.walkforward import walk_forward


@pytest.fixture(scope="module")
def prices():
    return synthetic.generate(600, seed=5)


def test_features_do_not_change_when_the_future_is_removed(prices):
    # Problem 2: features at day t must use data up to day t only.
    full = build_features(prices, window=30)
    for cut in (100, 250, 599):
        part = build_features(prices.iloc[:cut + 1], window=30)
        pd.testing.assert_frame_equal(full.iloc[:cut + 1], part, check_freq=False)


def test_changing_a_future_price_does_not_change_past_features(prices):
    changed = prices.copy()
    changed.loc[400:, ["open", "high", "low", "close"]] *= 3.0
    a, b = build_features(prices), build_features(changed)
    pd.testing.assert_frame_equal(a.iloc[:400], b.iloc[:400], check_freq=False)
    assert not a.iloc[400:].equals(b.iloc[400:])


def test_target_is_a_future_log_return(prices):
    # Problem 1: the target is r(t+h) with h >= 1, not today's price.
    y = make_target(prices, 7)
    c = prices["close"].to_numpy()
    assert y.iloc[10] == pytest.approx(np.log(c[17] / c[10]))
    assert y.iloc[-7:].isna().all() and not y.iloc[:-7].isna().any()
    with pytest.raises(ValueError):
        make_target(prices, 0)


def test_no_feature_is_the_target(prices):
    feats = build_features(prices)
    y1 = make_target(prices, 1)
    for col in feats.columns:
        both = pd.concat([feats[col].reset_index(drop=True), y1.reset_index(drop=True)], axis=1).dropna()
        assert abs(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]) < 0.5, col


def test_inequality_features_are_scale_free(prices):
    # Problem 7: absolute price differences drift with the price level. Returns do not.
    scaled = prices.copy()
    scaled[["open", "high", "low", "close"]] *= 1000
    a, b = build_features(prices), build_features(scaled)
    pd.testing.assert_frame_equal(a[INEQUALITY_FEATURES], b[INEQUALITY_FEATURES], check_freq=False)


def test_feature_sets():
    assert set(INEQUALITY_FEATURES).isdisjoint(BASE_FEATURES)
    assert FEATURE_SETS["base+inequality"] == BASE_FEATURES + INEQUALITY_FEATURES


@pytest.mark.parametrize("horizon", [1, 7])
def test_walk_forward_respects_the_embargo(prices, horizon):
    # Problems 3 and 8: a model trains only on targets observed before its first prediction.
    feats = build_features(prices)
    y = make_target(prices, horizon)
    res = walk_forward(feats, y, BASE_FEATURES, "ridge", horizon, min_train=200, refit_every=50)
    assert res.refits
    for r in res.refits:
        assert r.train_last + horizon <= r.origin
    # the same target series is used to train and to score
    pos = feats.index.get_indexer(res.dates)
    assert np.allclose(res.y_true, y.to_numpy()[pos])


def test_walk_forward_predictions_ignore_future_rows(prices):
    feats = build_features(prices)
    y = make_target(prices, 1)
    full = walk_forward(feats, y, BASE_FEATURES, "ridge", 1, min_train=200, refit_every=40)
    cut = 450
    part = walk_forward(feats.iloc[:cut], y.iloc[:cut], BASE_FEATURES, "ridge", 1, min_train=200, refit_every=40)
    n = part.y_pred.size
    assert np.allclose(full.y_pred[:n], part.y_pred)


def test_random_walk_predicts_zero(prices):
    feats = build_features(prices)
    res = walk_forward(feats, make_target(prices, 1), BASE_FEATURES, "random_walk", 1, min_train=200)
    assert np.all(res.y_pred == 0)


def test_walk_forward_needs_enough_rows(prices):
    feats = build_features(prices)
    with pytest.raises(ValueError):
        walk_forward(feats, make_target(prices, 1), BASE_FEATURES, "ridge", 1, min_train=5000)


def test_ar_model_uses_only_lagged_returns(prices):
    from btc_gini_signal.models import MODELS, columns_for
    assert columns_for(MODELS["ar"], FEATURE_SETS["base+inequality"]) == ["ret_lag1", "ret_lag2", "ret_lag3"]


def test_imputation_is_inside_the_pipeline():
    from btc_gini_signal.models import MODELS
    pipe = MODELS["ridge"].factory(0)
    assert [name for name, _ in pipe.steps] == ["impute", "scale", "model"]
