"""Efficient frontier construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.optimization import target_return_portfolio, minimum_variance_portfolio, maximum_sharpe_portfolio


def efficient_frontier(
    expected_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    n_points: int = 50,
    max_weight: float | None = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Build the efficient frontier by solving target-return problems
    across a grid of target returns.

    Returns DataFrame with columns: target_return, volatility, sharpe, weights (as dict).
    """
    if isinstance(expected_returns, pd.Series):
        er = expected_returns.values.astype(float)
    else:
        er = np.asarray(expected_returns, dtype=float)

    # Solve min-var and max-return to bound the frontier
    min_var = minimum_variance_portfolio(expected_returns, cov_matrix, max_weight=max_weight, risk_free_rate=risk_free_rate)
    # Upper bound: max achievable return is max(expected_returns)
    low = float(min_var["expected_return"])
    high = float(er.max())
    if high <= low:
        high = low + 0.01

    targets = np.linspace(low, high, n_points)
    records: list[dict] = []
    for t in targets:
        try:
            res = target_return_portfolio(expected_returns, cov_matrix, target_return=t, max_weight=max_weight, risk_free_rate=risk_free_rate)
        except RuntimeError:
            continue
        records.append(
            {
                "target_return": t,
                "expected_return": res["expected_return"],
                "volatility": res["volatility"],
                "sharpe": res["sharpe_ratio"],
                "weights": res["weights"].to_dict(),
            }
        )
    df = pd.DataFrame(records)
    return df
