"""Return-forecast metrics, the Diebold-Mariano test and a sign-strategy backtest with costs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2_oos(y, p) -> float:
    """Out-of-sample R^2 against the random walk (zero forecast): 1 - SSE(model) / SSE(zero)."""
    y, p = np.asarray(y), np.asarray(p)
    denom = np.sum(y ** 2)
    return float(1 - np.sum((y - p) ** 2) / denom) if denom > 0 else float("nan")


def directional_accuracy(y, p) -> float:
    """Share of rows where sign(prediction) = sign(return). Rows with a zero prediction or return are skipped."""
    y, p = np.asarray(y), np.asarray(p)
    mask = (y != 0) & (p != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y[mask]) == np.sign(p[mask])))


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_loss_diff: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def diebold_mariano(y, p_model, p_reference, horizon: int = 1, loss: str = "squared") -> DMResult:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

    d_t = loss(reference error) - loss(model error). A positive statistic means the model
    has a smaller loss than the reference. The long-run variance uses h-1 autocovariances,
    because h-step forecast errors overlap. The p-value is two-sided (t distribution, n-1 df).
    """
    y, a, b = np.asarray(y), np.asarray(p_model), np.asarray(p_reference)
    if loss == "squared":
        d = (y - b) ** 2 - (y - a) ** 2
    elif loss == "absolute":
        d = np.abs(y - b) - np.abs(y - a)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")
    n = d.size
    mean = d.mean()
    dc = d - mean
    gamma0 = np.dot(dc, dc) / n
    lrv = gamma0
    for k in range(1, horizon):
        lrv += 2 * np.dot(dc[k:], dc[:-k]) / n
    if lrv <= 0:
        lrv = gamma0
    if lrv == 0:
        return DMResult(0.0, 1.0, float(mean), n)
    dm = mean / np.sqrt(lrv / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(dm * correction)
    p_value = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return DMResult(stat, p_value, float(mean), n)


@dataclass
class BacktestResult:
    total_log_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    turnover: float
    hit_rate: float
    buy_hold_log_return: float
    buy_hold_sharpe: float

    def to_dict(self) -> dict:
        return asdict(self)


def _sharpe(r: np.ndarray, periods: int) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def _max_drawdown(log_returns: np.ndarray) -> float:
    equity = np.exp(np.cumsum(log_returns))
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float(np.min(equity / peak - 1))


def backtest_sign(y, p, cost_bps: float = 10.0, long_only: bool = False, periods: int = 365) -> BacktestResult:
    """Daily strategy for 1-day forecasts: hold sign(prediction) (or long/flat) for the next day.

    The cost is charged in basis points of notional for each unit change of the position.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    pos = (p > 0).astype(float) if long_only else np.sign(p)
    change = np.abs(np.diff(np.concatenate([[0.0], pos])))
    simple = np.expm1(y)
    strat_simple = pos * simple - change * cost_bps / 1e4
    strat = np.log1p(np.clip(strat_simple, -0.999999, None))
    active = pos != 0
    hit = float(np.mean(np.sign(y[active]) == pos[active])) if active.any() else float("nan")
    return BacktestResult(
        total_log_return=float(strat.sum()),
        annual_return=float(np.expm1(strat.mean() * periods)),
        annual_vol=float(strat.std(ddof=1) * np.sqrt(periods)),
        sharpe=_sharpe(strat, periods),
        max_drawdown=_max_drawdown(strat),
        turnover=float(change.mean()),
        hit_rate=hit,
        buy_hold_log_return=float(y.sum()),
        buy_hold_sharpe=_sharpe(y, periods),
    )


def summarize(y, p, y_naive=None, horizon: int = 1) -> dict:
    y_naive = np.zeros_like(np.asarray(y, dtype=float)) if y_naive is None else y_naive
    dm = diebold_mariano(y, p, y_naive, horizon)
    return {"n": int(np.size(y)), "mae": mae(y, p), "rmse": rmse(y, p), "r2_oos": r2_oos(y, p),
            "directional_accuracy": directional_accuracy(y, p), "dm_vs_random_walk": dm.to_dict()}
