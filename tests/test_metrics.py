import numpy as np
import pytest

from btc_gini_signal import metrics


def test_basic_errors():
    y = np.array([0.01, -0.02, 0.03])
    p = np.array([0.0, 0.0, 0.0])
    assert metrics.mae(y, p) == pytest.approx(0.02)
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((1e-4 + 4e-4 + 9e-4) / 3))
    assert metrics.r2_oos(y, p) == pytest.approx(0.0)  # the random walk is the reference
    assert metrics.r2_oos(y, y) == pytest.approx(1.0)


def test_directional_accuracy_skips_zeros():
    y = np.array([0.1, -0.1, 0.2, 0.0])
    p = np.array([0.5, 0.5, -0.1, 0.3])
    assert metrics.directional_accuracy(y, p) == pytest.approx(1 / 3)
    assert np.isnan(metrics.directional_accuracy(y, np.zeros(4)))


def test_dm_finds_a_better_forecast():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, 2000)
    y = signal + rng.normal(0, 1, 2000)
    good, bad = signal, np.zeros(2000)
    res = metrics.diebold_mariano(y, good, bad)
    assert res.statistic > 5 and res.p_value < 1e-6


def test_dm_equal_forecasts():
    y = np.random.default_rng(1).normal(size=300)
    res = metrics.diebold_mariano(y, np.zeros(300), np.zeros(300))
    assert res.statistic == 0 and res.p_value == 1.0


def test_dm_is_not_significant_for_noise():
    rng = np.random.default_rng(2)
    y = rng.normal(size=1500)
    res = metrics.diebold_mariano(y, rng.normal(0, 0.1, 1500), rng.normal(0, 0.1, 1500), horizon=7)
    assert res.p_value > 0.01


def test_dm_multi_step_variance_is_wider():
    rng = np.random.default_rng(3)
    e = np.convolve(rng.normal(size=1010), np.ones(7), "valid")[:1000]  # overlapping 7-step errors
    y = e + 0.3
    one = metrics.diebold_mariano(y, np.full(1000, 0.3), np.zeros(1000), horizon=1)
    seven = metrics.diebold_mariano(y, np.full(1000, 0.3), np.zeros(1000), horizon=7)
    assert abs(seven.statistic) < abs(one.statistic)


def test_dm_rejects_unknown_loss():
    with pytest.raises(ValueError):
        metrics.diebold_mariano([0.1], [0.0], [0.0], loss="cubic")


def test_backtest_costs_and_perfect_foresight():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    perfect = metrics.backtest_sign(y, y, cost_bps=0)
    assert perfect.total_log_return == pytest.approx(np.sum(np.log1p(np.abs(np.expm1(y)))))
    with_cost = metrics.backtest_sign(y, y, cost_bps=10)
    assert with_cost.total_log_return < perfect.total_log_return
    assert perfect.hit_rate == 1.0 and perfect.max_drawdown == pytest.approx(0.0)


def test_backtest_long_only_equals_buy_and_hold_when_always_long():
    y = np.random.default_rng(4).normal(0, 0.02, 200)
    bt = metrics.backtest_sign(y, np.ones(200), cost_bps=0, long_only=True)
    assert bt.total_log_return == pytest.approx(y.sum())
    assert bt.turnover == pytest.approx(1 / 200)


def test_summarize_keys():
    y = np.random.default_rng(5).normal(size=100)
    s = metrics.summarize(y, y * 0.1)
    assert {"mae", "rmse", "r2_oos", "directional_accuracy", "dm_vs_random_walk"} <= set(s)
