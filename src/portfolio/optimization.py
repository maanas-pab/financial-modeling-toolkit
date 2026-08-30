"""Portfolio optimization via scipy.optimize."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _validate_weights(weights: np.ndarray, tol: float = 1e-6) -> None:
    if not np.all(weights >= -1e-8):
        raise ValueError("Weights contain negative values beyond tolerance")
    if abs(weights.sum() - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1; got {weights.sum()}")


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    return float(np.dot(weights, expected_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov_matrix @ weights))


def portfolio_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0.0
    return (ret - risk_free_rate) / vol


def _prepare_inputs(
    expected_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if isinstance(expected_returns, pd.Series):
        labels = expected_returns.index.tolist()
        er = expected_returns.values.astype(float)
    else:
        er = np.asarray(expected_returns, dtype=float)
        labels = [f"Asset {i}" for i in range(len(er))]
    if isinstance(cov_matrix, pd.DataFrame):
        cov = cov_matrix.values.astype(float)
    else:
        cov = np.asarray(cov_matrix, dtype=float)
    if er.ndim != 1:
        raise ValueError("expected_returns must be 1-d")
    if cov.shape[0] != cov.shape[1] or cov.shape[0] != len(er):
        raise ValueError("cov_matrix shape mismatch with expected_returns")
    if len(er) < 2:
        raise ValueError("Need at least 2 assets")
    return er, cov, labels


def minimum_variance_portfolio(
    expected_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    max_weight: float | None = None,
    risk_free_rate: float = 0.0,
    transaction_cost: float | np.ndarray | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_limit: float | None = None,
) -> dict:
    """
    Global minimum variance portfolio.

    Minimizes portfolio variance subject to sum(w)=1 and optional max weight.
    Supports transaction-cost adjustment (expected return net of costs) and
    turnover constraints (|w - w_prev|_1 <= limit).
    """
    er, cov, labels = _prepare_inputs(expected_returns, cov_matrix)
    n = len(er)
    if transaction_cost is not None:
        if np.isscalar(transaction_cost):
            er = er - float(transaction_cost)
        else:
            tc = np.asarray(transaction_cost, dtype=float)
            if tc.shape[0] != n:
                raise ValueError("transaction_cost length mismatch")
            er = er - tc
    if max_weight is not None and max_weight * n < 1 - 1e-8:
        raise ValueError(f"max_weight {max_weight} infeasible: n*max_weight < 1")
    if turnover_limit is not None:
        if prev_weights is None:
            raise ValueError("turnover_limit requires prev_weights")
        if turnover_limit < 0:
            raise ValueError("turnover_limit cannot be negative")
        prev_weights = np.asarray(prev_weights, dtype=float)

    bounds = [(0, max_weight) if max_weight is not None else (0, 1) for _ in range(n)]
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if turnover_limit is not None:
        constraints.append({"type": "ineq", "fun": lambda w: turnover_limit - np.sum(np.abs(w - prev_weights))})
    x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    ret = portfolio_return(w, er)
    vol = portfolio_volatility(w, cov)
    sharpe = portfolio_sharpe(w, er, cov, risk_free_rate)
    return {
        "weights": pd.Series(w, index=labels),
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }


def maximum_sharpe_portfolio(
    expected_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    risk_free_rate: float = 0.0,
    max_weight: float | None = None,
    transaction_cost: float | np.ndarray | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_limit: float | None = None,
) -> dict:
    """
    Maximum Sharpe ratio portfolio (tangency portfolio). Supports
    transaction-cost and turnover constraints (see minimum_variance).
    """
    er, cov, labels = _prepare_inputs(expected_returns, cov_matrix)
    n = len(er)
    if transaction_cost is not None:
        if np.isscalar(transaction_cost):
            er = er - float(transaction_cost)
        else:
            tc = np.asarray(transaction_cost, dtype=float)
            if tc.shape[0] != n:
                raise ValueError("transaction_cost length mismatch")
            er = er - tc
    if max_weight is not None and max_weight * n < 1 - 1e-8:
        raise ValueError(f"max_weight {max_weight} infeasible: n*max_weight < 1")
    if turnover_limit is not None:
        if prev_weights is None:
            raise ValueError("turnover_limit requires prev_weights")
        if turnover_limit < 0:
            raise ValueError("turnover_limit cannot be negative")
        prev_weights = np.asarray(prev_weights, dtype=float)

    bounds = [(0, max_weight) if max_weight is not None else (0, 1) for _ in range(n)]
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if turnover_limit is not None:
        constraints.append({"type": "ineq", "fun": lambda w: turnover_limit - np.sum(np.abs(w - prev_weights))})
    x0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        vol = portfolio_volatility(w, cov)
        if vol == 0:
            return 1e6
        ret = portfolio_return(w, er)
        return -(ret - risk_free_rate) / vol

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    ret = portfolio_return(w, er)
    vol = portfolio_volatility(w, cov)
    sharpe = portfolio_sharpe(w, er, cov, risk_free_rate)
    return {
        "weights": pd.Series(w, index=labels),
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }


def target_return_portfolio(
    expected_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    target_return: float,
    max_weight: float | None = None,
    risk_free_rate: float = 0.0,
    transaction_cost: float | np.ndarray | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_limit: float | None = None,
) -> dict:
    """
    Minimum volatility portfolio achieving at least target_return.
    Supports transaction-cost (net return) and turnover constraints.
    """
    er, cov, labels = _prepare_inputs(expected_returns, cov_matrix)
    n = len(er)
    if transaction_cost is not None:
        if np.isscalar(transaction_cost):
            er = er - float(transaction_cost)
        else:
            tc = np.asarray(transaction_cost, dtype=float)
            if tc.shape[0] != n:
                raise ValueError("transaction_cost length mismatch")
            er = er - tc
    if turnover_limit is not None:
        if prev_weights is None:
            raise ValueError("turnover_limit requires prev_weights")
        if turnover_limit < 0:
            raise ValueError("turnover_limit cannot be negative")
        prev_weights = np.asarray(prev_weights, dtype=float)
    if target_return < er.min() - 1e-8 or target_return > er.max() + 1e-8:
        # Still solve; optimizer may be infeasible but we try
        pass
    if max_weight is not None and max_weight * n < 1 - 1e-8:
        raise ValueError(f"max_weight {max_weight} infeasible")

    bounds = [(0, max_weight) if max_weight is not None else (0, 1) for _ in range(n)]
    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: portfolio_return(w, er) - target_return},
    ]
    if turnover_limit is not None:
        constraints.append({"type": "ineq", "fun": lambda w: turnover_limit - np.sum(np.abs(w - prev_weights))})
    x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Target return optimization failed (target={target_return}): {res.message}")

    w = res.x
    ret = portfolio_return(w, er)
    vol = portfolio_volatility(w, cov)
    sharpe = portfolio_sharpe(w, er, cov, risk_free_rate)
    return {
        "weights": pd.Series(w, index=labels),
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }


def monte_carlo_simulation(
    expected_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate random portfolios and compute risk-return metrics.
    Reproducible via fixed seed. Transaction-cost and turnover-aware
    extensions can be added via `expected_returns` net of costs.
    """
    er, cov, labels = _prepare_inputs(expected_returns, cov_matrix)
    n = len(er)
    rng = np.random.default_rng(seed)
    weights = rng.random((n_portfolios, n))
    weights = weights / weights.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rets = weights @ er
        vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
        # Avoid divide-by-zero for degenerate cov
        vols = np.where(vols == 0, np.nan, vols)
        sharpes = (rets - risk_free_rate) / vols

    df = pd.DataFrame({"return": rets, "volatility": vols, "sharpe": sharpes})
    return df
