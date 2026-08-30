"""Tests for portfolio returns, risk, and optimization."""

import pytest
import numpy as np
import pandas as pd

from src.portfolio.returns import simple_returns, log_returns, cumulative_returns, annualized_return, annualized_volatility
from src.portfolio.risk import sharpe_ratio, sortino_ratio, max_drawdown, correlation_matrix, covariance_matrix
from src.portfolio.optimization import (
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    target_return_portfolio,
    monte_carlo_simulation,
)
from src.portfolio.efficient_frontier import efficient_frontier


@pytest.fixture
def prices():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(0)
    # 3 assets lognormal-like
    a = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 100))
    b = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, 100))
    c = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.008, 100))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


@pytest.fixture
def returns(prices):
    return simple_returns(prices)


def test_simple_returns_valid(prices):
    r = simple_returns(prices)
    assert not r.isnull().any().any()
    assert len(r) == len(prices) - 1

def test_simple_returns_negative_price_raises():
    df = pd.DataFrame({"A": [100, -1, 101]})
    with pytest.raises(ValueError):
        simple_returns(df)

def test_log_returns(prices):
    lr = log_returns(prices)
    assert len(lr) == len(prices) - 1
    # log return approx equal to simple for small moves? just check negative values possible
    assert isinstance(lr, pd.DataFrame)

def test_cumulative_returns(returns):
    cum = cumulative_returns(returns)
    assert cum.shape == returns.shape
    # last value should equal (prod(1+r)-1) per column
    for col in returns.columns:
        expected = float((1 + returns[col]).prod() - 1)
        assert abs(cum[col].iloc[-1] - expected) < 1e-8

def test_annualized_return_series(returns):
    ar = annualized_return(returns["A"])
    assert isinstance(ar, float)
    # Should be reasonable: small daily returns => annualized maybe +/- 20%
    assert -0.9 < ar < 2.0

def test_annualized_return_dataframe(returns):
    ar = annualized_return(returns)
    assert isinstance(ar, pd.Series)
    assert len(ar) == 3

def test_sharpe_ratio_series(returns):
    sr = sharpe_ratio(returns["A"])
    assert isinstance(sr, float)

def test_sharpe_ratio_dataframe(returns):
    sr = sharpe_ratio(returns)
    assert isinstance(sr, pd.Series)
    assert len(sr) == 3

def test_sortino_ratio(returns):
    sr = sortino_ratio(returns["A"])
    assert isinstance(sr, float)

def test_max_drawdown_prices(prices):
    dd = max_drawdown(prices=prices["A"])
    assert dd <= 0
    assert dd > -1

def test_max_drawdown_returns(returns):
    dd = max_drawdown(returns=returns["A"])
    assert dd <= 0

def test_correlation_matrix(returns):
    corr = correlation_matrix(returns)
    assert corr.shape == (3, 3)
    # diagonal 1
    assert np.allclose(np.diag(corr.values), 1.0)
    assert (corr.values >= -1).all() and (corr.values <= 1).all()

def test_covariance_matrix_annualized(returns):
    cov = covariance_matrix(returns, annualize=True)
    assert cov.shape == (3, 3)
    # Annualized larger than non-annualized
    cov2 = covariance_matrix(returns, annualize=False)
    assert (cov.values > cov2.values).all() or (cov.values >= cov2.values).all()

# ---------- Optimization ----------

def _prep_expected(returns):
    er = annualized_return(returns)  # Series
    cov = covariance_matrix(returns)
    return er, cov

def test_min_variance_weights_sum_to_one(returns):
    er, cov = _prep_expected(returns)
    res = minimum_variance_portfolio(er, cov)
    assert abs(res["weights"].sum() - 1) < 1e-6
    assert (res["weights"] >= -1e-8).all()
    assert res["volatility"] > 0

def test_min_variance_max_weight_constraint(returns):
    er, cov = _prep_expected(returns)
    res = minimum_variance_portfolio(er, cov, max_weight=0.6)
    assert res["weights"].max() <= 0.6 + 1e-6

def test_max_sharpe_weights_sum_to_one(returns):
    er, cov = _prep_expected(returns)
    res = maximum_sharpe_portfolio(er, cov, risk_free_rate=0.02)
    assert abs(res["weights"].sum() - 1) < 1e-6

def test_target_return_portfolio(returns):
    er, cov = _prep_expected(returns)
    target = float(er.mean())
    res = target_return_portfolio(er, cov, target_return=target)
    assert abs(res["weights"].sum() - 1) < 1e-6
    assert res["expected_return"] >= target - 1e-4

def test_max_weight_infeasible_raises(returns):
    er, cov = _prep_expected(returns)
    with pytest.raises(ValueError):
        minimum_variance_portfolio(er, cov, max_weight=0.1)  # 3*0.1=0.3 <1

def test_monte_carlo_shape(returns):
    er, cov = _prep_expected(returns)
    df = monte_carlo_simulation(er, cov, n_portfolios=500, seed=42)
    assert df.shape[0] == 500
    assert set(df.columns) == {"return", "volatility", "sharpe"}
    # Reproducibility
    df2 = monte_carlo_simulation(er, cov, n_portfolios=500, seed=42)
    assert np.allclose(df["return"].values, df2["return"].values)

def test_efficient_frontier_shape(returns):
    er, cov = _prep_expected(returns)
    front = efficient_frontier(er, cov, n_points=10)
    assert len(front) > 0
    assert "volatility" in front.columns
    assert "expected_return" in front.columns
    # Volatility non-decreasing? frontier moves up-right generally
    # just check sorted by volatility increasing roughly
    assert front["volatility"].iloc[0] <= front["volatility"].iloc[-1] + 1e-6
