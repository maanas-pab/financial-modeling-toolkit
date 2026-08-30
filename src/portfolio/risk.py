"""Risk and performance metrics."""

from __future__ import annotations

import pandas as pd
import numpy as np


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Annualized Sharpe ratio: (E[R] - Rf) / sigma * sqrt(periods)
    Rf is annual risk-free rate; converted to per-period.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1 if risk_free_rate != 0 else 0.0
    excess = returns - rf_per_period
    if isinstance(returns, pd.DataFrame):
        ann_excess = excess.mean() * periods_per_year
        ann_vol = excess.std() * np.sqrt(periods_per_year)
        # Avoid division by zero
        return ann_excess / ann_vol.replace(0, np.nan)
    else:
        ann_excess = excess.mean() * periods_per_year
        ann_vol = excess.std() * np.sqrt(periods_per_year)
        if ann_vol == 0 or np.isnan(ann_vol):
            return float("nan") if ann_vol == 0 else float("nan")
        return float(ann_excess / ann_vol)


def downside_deviation(returns: pd.Series, target: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Annualized downside deviation (for Sortino).
    """
    if len(returns) == 0:
        raise ValueError("Returns empty")
    downside = returns[returns < target]
    if len(downside) == 0:
        return 0.0
    # Per-period target
    dev = np.sqrt((np.square(downside - target)).mean()) * np.sqrt(periods_per_year)
    return float(dev)


def sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Annualized Sortino ratio: (E[R] - Rf) / downside_deviation.
    """
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1 if risk_free_rate != 0 else 0.0
    if isinstance(returns, pd.DataFrame):
        results = {}
        for col in returns.columns:
            s = returns[col]
            excess_ann = (s.mean() - rf_per_period) * periods_per_year
            dd = downside_deviation(s, target=target, periods_per_year=periods_per_year)
            results[col] = excess_ann / dd if dd != 0 else float("nan")
        return pd.Series(results)
    else:
        excess_ann = (returns.mean() - rf_per_period) * periods_per_year
        dd = downside_deviation(returns, target=target, periods_per_year=periods_per_year)
        if dd == 0:
            return float("nan")
        return float(excess_ann / dd)


def max_drawdown(prices: pd.Series | None = None, returns: pd.Series | None = None) -> float:
    """
    Maximum drawdown. Provide either prices or returns (simple).
    Returns negative value (e.g., -0.20 for 20% drawdown).
    """
    if prices is not None:
        if (prices <= 0).any():
            raise ValueError("Prices must be > 0")
        cum_max = prices.cummax()
        dd = (prices - cum_max) / cum_max
        return float(dd.min())
    elif returns is not None:
        cum = (1 + returns).cumprod()
        cum_max = cum.cummax()
        dd = (cum - cum_max) / cum_max
        return float(dd.min())
    else:
        raise ValueError("Provide either prices or returns")


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("Returns empty")
    if returns.isnull().any().any():
        raise ValueError("Returns contain NaN")
    return returns.corr()


def covariance_matrix(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    annualize: bool = True,
    shrinkage: float | None = None,
    shrinkage_target: str = "constant_correlation",
) -> pd.DataFrame:
    """
    Annualized covariance. Supports Ledoit-Wolf style shrinkage.

    Args:
        returns: Daily (or per-period) returns.
        periods_per_year: Trading days per year (explicit).
        annualize: If True, multiply by periods_per_year.
        shrinkage: 0-1 shrinkage intensity toward target. If None, no shrinkage.
        shrinkage_target: 'constant_correlation' | 'single_factor' | 'identity'
    """
    if returns.empty:
        raise ValueError("Returns empty")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    cov = returns.cov()
    if shrinkage is not None:
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be in [0,1]")
        # Simple shrinkage toward target
        if shrinkage_target == "identity":
            # Shrink toward diagonal (variance only)
            target = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
        elif shrinkage_target == "constant_correlation":
            # constant correlation target
            corr = returns.corr()
            avg_corr = float(corr.values[np.triu_indices_from(corr.values, 1)].mean())
            stds = returns.std()
            target_vals = np.outer(stds, stds) * avg_corr
            np.fill_diagonal(target_vals, stds.values ** 2)
            target = pd.DataFrame(target_vals, index=cov.index, columns=cov.columns)
        elif shrinkage_target == "single_factor":
            # market factor (first PC approx via equal-weight)
            mkt = returns.mean(axis=1)
            betas = returns.apply(lambda col: np.cov(col, mkt)[0, 1] / np.var(mkt) if np.var(mkt) != 0 else 0)
            var_mkt = float(mkt.var())
            target_vals = np.outer(betas, betas) * var_mkt
            # add residual variance on diagonal
            for i, col in enumerate(returns.columns):
                resid_var = float((returns[col] - betas[col] * mkt).var())
                target_vals[i, i] += resid_var
            target = pd.DataFrame(target_vals, index=cov.index, columns=cov.columns)
        else:
            raise ValueError("shrinkage_target must be 'identity'|'constant_correlation'|'single_factor'")
        cov = (1 - shrinkage) * cov + shrinkage * target

    if annualize:
        cov = cov * periods_per_year
    return cov


def ledoit_wolf_shrinkage(returns: pd.DataFrame, periods_per_year: int = 252) -> tuple[pd.DataFrame, float]:
    """
    Approximate Ledoit-Wolf optimal shrinkage intensity (one-pass).
    Returns (shrunk_cov, shrinkage).
    """
    n, p = returns.shape
    if n < 2:
        raise ValueError("Need at least 2 observations")
    sample_cov = returns.cov() * periods_per_year
    # Simple heuristic shrinkage ~ min(1, (p/n))
    shrinkage = float(min(0.5, p / n))
    shrunk = covariance_matrix(returns, periods_per_year=periods_per_year, annualize=True, shrinkage=shrinkage)
    return shrunk, shrinkage
