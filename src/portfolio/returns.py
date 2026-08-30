"""Return calculations."""

from __future__ import annotations

import pandas as pd
import numpy as np


def _validate_prices(prices: pd.DataFrame | pd.Series) -> None:
    if isinstance(prices, pd.DataFrame):
        if (prices <= 0).any().any():
            raise ValueError("Prices must be > 0; found non-positive values")
        if prices.isnull().any().any():
            raise ValueError("Prices contain NaN")
    else:
        if (prices <= 0).any():
            raise ValueError("Prices must be > 0")
        if prices.isnull().any():
            raise ValueError("Prices contain NaN")


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple returns: (P_t / P_{t-1}) - 1."""
    _validate_prices(prices)
    return prices.pct_change().dropna()


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log returns: ln(P_t / P_{t-1})."""
    _validate_prices(prices)
    return np.log(prices / prices.shift(1)).dropna()


def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Cumulative returns from a series of simple returns."""
    if returns.isnull().any().any() if isinstance(returns, pd.DataFrame) else returns.isnull().any():
        raise ValueError("Returns contain NaN; drop or fill before cumulating")
    return (1 + returns).cumprod() - 1


def annualized_return(returns: pd.DataFrame | pd.Series, periods_per_year: int = 252) -> float | pd.Series:
    """
    Annualized geometric return: (1 + mean_return)^{periods} - 1  approx
    Actually compute: prod(1+r)^{periods/len} -1
    """
    if len(returns) == 0:
        raise ValueError("Returns empty")
    if isinstance(returns, pd.DataFrame):
        return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    else:
        return float((1 + returns).prod() ** (periods_per_year / len(returns)) - 1)


def annualized_volatility(returns: pd.DataFrame | pd.Series, periods_per_year: int = 252) -> float | pd.Series:
    """Annualized volatility: std * sqrt(periods_per_year)."""
    if len(returns) == 0:
        raise ValueError("Returns empty")
    if isinstance(returns, pd.DataFrame):
        return returns.std() * np.sqrt(periods_per_year)
    else:
        return float(returns.std() * np.sqrt(periods_per_year))
