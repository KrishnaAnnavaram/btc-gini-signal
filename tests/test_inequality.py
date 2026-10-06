import numpy as np
import pytest
from scipy.optimize import brentq

from btc_gini_signal.features import gini, kolkata, lorenz_curve, rolling_gini_kolkata


def test_equal_values_give_zero_gini_and_half_kolkata():
    assert gini([3, 3, 3, 3]) == pytest.approx(0.0)
    assert kolkata([3, 3, 3, 3]) == pytest.approx(0.5)


def test_one_holder_gives_maximum_values():
    x = [0, 0, 0, 0, 10]
    assert gini(x) == pytest.approx(4 / 5)  # (n-1)/n for a finite sample
    assert kolkata(x) == pytest.approx(5 / 6, abs=1e-9)  # f crosses 0 between p=0.8 and p=1


def test_known_two_point_values():
    # values 1 and 3: mean abs difference = 1, mean = 2 -> G = 1 / (2 * 2) = 0.25
    assert gini([1, 3]) == pytest.approx(0.25)


def test_exponential_distribution_matches_theory():
    # Exponential: Gini = 0.5 and L(p) = p + (1 - p) ln(1 - p)
    x = np.random.default_rng(0).exponential(size=200_000)
    assert gini(x) == pytest.approx(0.5, abs=0.01)
    k_theory = brentq(lambda p: p + (1 - p) * np.log(1 - p) - (1 - p), 0.5, 0.99)
    assert kolkata(x) == pytest.approx(k_theory, abs=0.01)


def test_uniform_distribution_gini_is_one_third():
    x = np.random.default_rng(1).uniform(size=200_000)
    assert gini(x) == pytest.approx(1 / 3, abs=0.01)


def test_indices_are_scale_free():
    x = np.random.default_rng(2).exponential(size=500)
    assert gini(x * 1000) == pytest.approx(gini(x))
    assert kolkata(x * 1000) == pytest.approx(kolkata(x))


def test_lorenz_curve_shape():
    p, L = lorenz_curve([1, 2, 3, 4])
    assert p[0] == 0 and L[0] == 0 and p[-1] == 1 and L[-1] == pytest.approx(1)
    assert np.all(L <= p + 1e-12) and np.all(np.diff(L) >= 0)


def test_rejects_negative_or_empty_input():
    with pytest.raises(ValueError):
        gini([-1, 2])
    with pytest.raises(ValueError):
        kolkata([])


def test_rolling_matches_single_sample_functions():
    x = np.abs(np.random.default_rng(3).standard_t(3, size=120))
    g, k = rolling_gini_kolkata(x, 30)
    assert np.isnan(g[:29]).all() and np.isnan(k[:29]).all()
    for t in (29, 60, 119):
        assert g[t] == pytest.approx(gini(x[t - 29:t + 1]))
        assert k[t] == pytest.approx(kolkata(x[t - 29:t + 1]))


def test_rolling_window_with_nan_stays_nan():
    x = np.ones(40)
    x[10] = np.nan
    g, _ = rolling_gini_kolkata(x, 5)
    assert np.isnan(g[10:15]).all() and g[20] == pytest.approx(0.0)
