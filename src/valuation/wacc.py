"""WACC and cost-of-capital calculations."""

from __future__ import annotations


def calculate_cost_of_equity(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    country_risk_premium: float = 0.0,
    alpha: float = 0.0,
) -> float:
    """
    CAPM: Ke = Rf + Beta * ERP + CRP + alpha.

    Args:
        risk_free_rate: Risk-free rate (decimal, e.g., 0.03 for 3%).
        beta: Equity beta (levered).
        equity_risk_premium: Equity risk premium (ERP).
        country_risk_premium: Additional CRP for emerging markets (default 0).
        alpha: Company-specific alpha / Jensen's alpha (default 0).

    Returns:
        Cost of equity as decimal.
    """
    if not -0.5 <= risk_free_rate <= 0.30:
        raise ValueError(f"risk_free_rate {risk_free_rate} looks implausible; expected decimal like 0.03")
    if beta < 0:
        raise ValueError("beta cannot be negative")
    if equity_risk_premium < 0 or equity_risk_premium > 0.30:
        raise ValueError(f"equity_risk_premium {equity_risk_premium} outside [0, 0.30]")
    if not 0 <= country_risk_premium <= 0.30:
        raise ValueError(f"country_risk_premium {country_risk_premium} outside [0, 0.30]")
    return risk_free_rate + beta * equity_risk_premium + country_risk_premium + alpha


def calculate_cost_of_equity_ff3(
    risk_free_rate: float,
    beta_market: float,
    equity_risk_premium: float,
    beta_size: float = 0.0,
    size_premium: float = 0.0,
    beta_value: float = 0.0,
    value_premium: float = 0.0,
    country_risk_premium: float = 0.0,
) -> float:
    """
    Fama-French 3-factor cost of equity: Rf + b_mkt*ERP + b_smb*SMB + b_hml*HML + CRP.
    Falls back to CAPM when size/value betas are zero.
    """
    for v, name in [(beta_market, "beta_market"), (beta_size, "beta_size"), (beta_value, "beta_value")]:
        if v < 0:
            raise ValueError(f"{name} cannot be negative")
    return risk_free_rate + beta_market * equity_risk_premium + beta_size * size_premium + beta_value * value_premium + country_risk_premium


def unlever_beta(levered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    """Unlever: beta_asset = beta_equity / (1 + (1-T)*D/E)."""
    if debt_to_equity < 0:
        raise ValueError("debt_to_equity cannot be negative")
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0,1)")
    return levered_beta / (1 + (1 - tax_rate) * debt_to_equity)


def lever_beta(unlevered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    """Re-lever: beta_equity = beta_asset * (1 + (1-T)*D/E)."""
    if debt_to_equity < 0:
        raise ValueError("debt_to_equity cannot be negative")
    return unlevered_beta * (1 + (1 - tax_rate) * debt_to_equity)


def calculate_after_tax_cost_of_debt(
    cost_of_debt: float,
    tax_rate: float,
) -> float:
    """
    After-tax cost of debt: Kd * (1 - TaxRate).
    """
    if cost_of_debt < 0 or cost_of_debt > 0.50:
        raise ValueError(f"cost_of_debt {cost_of_debt} outside plausible [0, 0.50] range")
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    return cost_of_debt * (1 - tax_rate)


def calculate_wacc(
    market_cap: float,
    debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
    cash: float = 0.0,
) -> float:
    """
    WACC = (E/V * Ke) + (D/V * Kd * (1 - Tc))

    Uses enterprise value V = E + D (optionally net of cash is NOT subtracted
    here; pass gross debt). For net debt treatment, subtract cash from D before calling.

    Args:
        market_cap: Market value of equity.
        debt: Market value of debt (or book if market unavailable).
        cost_of_equity: Ke as decimal.
        cost_of_debt: Pre-tax cost of debt as decimal.
        tax_rate: Corporate tax rate.
        cash: Unused; kept for interface symmetry, not part of WACC weighting.

    Returns:
        WACC as decimal.
    """
    if market_cap < 0 or debt < 0:
        raise ValueError("market_cap and debt must be >= 0")
    if market_cap == 0 and debt == 0:
        raise ValueError("Both market_cap and debt are zero; WACC undefined")
    if not 0 <= cost_of_equity <= 0.50:
        raise ValueError(f"cost_of_equity {cost_of_equity} outside [0, 0.50]")
    if not 0 <= cost_of_debt <= 0.50:
        raise ValueError(f"cost_of_debt {cost_of_debt} outside [0, 0.50]")
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")

    total = market_cap + debt
    w_e = market_cap / total
    w_d = debt / total
    kd_after = calculate_after_tax_cost_of_debt(cost_of_debt, tax_rate)
    wacc = w_e * cost_of_equity + w_d * kd_after
    if wacc <= 0:
        raise ValueError(f"Computed WACC {wacc} is not positive; check inputs")
    if wacc > 0.50:
        raise ValueError(f"Computed WACC {wacc:.1%} is implausibly high (>50%)")
    return wacc


def wacc_from_components(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    cost_of_debt: float,
    tax_rate: float,
    market_cap: float,
    debt: float,
) -> dict[str, float]:
    """
    Convenience: compute Ke, after-tax Kd, and WACC in one call.
    Returns dict with ke, kd_after_tax, wacc.
    """
    ke = calculate_cost_of_equity(risk_free_rate, beta, equity_risk_premium)
    kd_at = calculate_after_tax_cost_of_debt(cost_of_debt, tax_rate)
    wacc = calculate_wacc(market_cap, debt, ke, cost_of_debt, tax_rate)
    return {"ke": ke, "kd_after_tax": kd_at, "wacc": wacc}
