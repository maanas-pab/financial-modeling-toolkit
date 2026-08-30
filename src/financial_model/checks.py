"""Automated integrity checks for the three-statement model."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.financial_model.three_statement import ThreeStatementModel


def check_balance_sheet(model: ThreeStatementModel, tol: float = 1e-6) -> pd.DataFrame:
    """
    Check that Assets == Liabilities + Equity (including leases/minority where present).

    Uses Total Liab. & Equity as definitive plug check, with A-L-E breakdown.
    Returns a DataFrame with pass/fail per period.
    """
    assert model.balance_sheet is not None
    bs = model.balance_sheet
    assets = bs.loc["Total Assets"]
    liab = bs.loc["Total Liabilities"]
    equity = bs.loc["Equity"]
    # Total Liab. & Equity is source of truth (includes minority/lease)
    total_le = bs.loc["Total Liab. & Equity"] if "Total Liab. & Equity" in bs.index else liab + equity
    diff_total = assets - total_le
    # Legacy A-L-E for backwards compat (now includes lease in liab, minority separate)
    minority = bs.loc["Minority Interest"] if "Minority Interest" in bs.index else 0
    # if minority is scalar 0, broadcast
    if isinstance(minority, (int, float)):
        diff_ale = diff_total
    else:
        diff_ale = assets - liab - equity - minority
        # Prefer total check for pass/fail
        diff_ale = diff_total
    result = pd.DataFrame(
        {
            "Assets": assets,
            "Liabilities": liab,
            "Equity": equity,
            "Total L+E": total_le,
            "Diff (A - L-E)": diff_total,
            "Pass": diff_total.abs() < tol,
        }
    )
    return result


def check_cash_reconciliation(model: ThreeStatementModel, tol: float = 1e-6) -> pd.DataFrame:
    """
    Verify that cash on balance sheet equals ending cash from CFS and
    that net change in cash reconciles to OCF+ICF+FCF.
    """
    assert model.balance_sheet is not None
    assert model.cash_flow is not None
    bs_cash = model.balance_sheet.loc["Cash"]
    cf_end = model.cash_flow.loc["Ending Cash"]
    diff = bs_cash - cf_end

    # Also reconcile net change
    cf_net = model.cash_flow.loc["Net Change in Cash"]
    ocf = model.cash_flow.loc["Operating Cash Flow"]
    icf = model.cash_flow.loc["Investing Cash Flow"]
    fcf = model.cash_flow.loc["Financing Cash Flow"]
    recon = ocf + icf + fcf
    diff2 = cf_net - recon

    df = pd.DataFrame(
        {
            "BS Cash": bs_cash,
            "CFS Ending Cash": cf_end,
            "Cash Diff": diff,
            "Cash Pass": diff.abs() < tol,
            "CFS Net Change": cf_net,
            "OCF+ICF+FCF": recon,
            "Recon Diff": diff2,
            "Recon Pass": diff2.abs() < tol,
        }
    )
    return df


def check_debt_reconciliation(model: ThreeStatementModel, tol: float = 1e-6) -> pd.DataFrame:
    """
    Verify debt continuity: Debt_t = Debt_{t-1} + Issuance - Repayment.
    """
    assert model.balance_sheet is not None
    debt = model.balance_sheet.loc["Debt"]
    cf_issue = model.cash_flow.loc["Debt Issuance"]
    cf_repay = model.cash_flow.loc["Debt Repayment"]  # negative
    # cf_repay is negative, so net = issue + repay
    a = model.assumptions
    debt_prev = a.beginning_debt
    records = []
    for period in debt.index:
        expected = debt_prev + cf_issue[period] + cf_repay[period]
        actual = debt[period]
        records.append(
            {
                "Actual Debt": actual,
                "Expected Debt": expected,
                "Diff": actual - expected,
                "Pass": abs(actual - expected) < tol,
            }
        )
        debt_prev = actual
    return pd.DataFrame(records, index=debt.index)


def check_retained_earnings(model: ThreeStatementModel, tol: float = 1e-6) -> pd.DataFrame:
    """
    Verify RE continuity: RE_t = RE_{t-1} + Net Income - Dividends.
    """
    assert model.balance_sheet is not None
    assert model.cash_flow is not None
    re = model.balance_sheet.loc["Retained Earnings (memo)"]
    ni = model.cash_flow.loc["Net Income"]
    dividends = -model.cash_flow.loc["Dividends"]  # positive value
    a = model.assumptions
    re_prev = a.beginning_retained_earnings
    records = []
    for period in re.index:
        expected = re_prev + ni[period] - dividends[period]
        actual = re[period]
        records.append(
            {
                "Actual RE": actual,
                "Expected RE": expected,
                "Diff": actual - expected,
                "Pass": abs(actual - expected) < tol,
            }
        )
        re_prev = actual
    return pd.DataFrame(records, index=re.index)


def run_all_checks(model: ThreeStatementModel, tol: float = 1e-6) -> dict[str, pd.DataFrame]:
    """
    Run all integrity checks and return a dict of DataFrames.

    Also adds a summary row indicating overall pass/fail.
    """
    results = {
        "balance_sheet": check_balance_sheet(model, tol),
        "cash": check_cash_reconciliation(model, tol),
        "debt": check_debt_reconciliation(model, tol),
        "retained_earnings": check_retained_earnings(model, tol),
    }
    return results


def is_balanced(model: ThreeStatementModel, tol: float = 1e-6) -> bool:
    """Return True if all checks pass."""
    results = run_all_checks(model, tol)
    for df in results.values():
        # Look for any Pass column that is False
        pass_cols = [c for c in df.columns if "Pass" in c]
        for c in pass_cols:
            if not df[c].all():
                return False
    return True
