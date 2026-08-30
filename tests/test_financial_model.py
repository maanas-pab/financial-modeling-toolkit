"""Tests for three-statement model, integrity checks, and scenario engine."""

import pytest
import pandas as pd

from src.financial_model.assumptions import ModelAssumptions, ScenarioAssumptions, BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO
from src.financial_model.three_statement import ThreeStatementModel
from src.financial_model.checks import run_all_checks, is_balanced
from src.valuation.sensitivity import scenario_comparison


def _base_assumptions():
    return ModelAssumptions(
        historical_revenue=500,
        revenue_growth=[0.06, 0.05, 0.04, 0.03, 0.03],
        gross_margin=0.40,
        opex_margin=0.20,
        da_pct_revenue=0.03,
        capex_pct_revenue=0.04,
        tax_rate=0.25,
        ar_days=45,
        inventory_days=30,
        ap_days=40,
        beginning_cash=80,
        beginning_ar=62,
        beginning_inventory=25,
        beginning_ppe=200,
        beginning_other_assets=20,
        beginning_ap=30,
        beginning_debt=150,
        beginning_other_liabilities=10,
        beginning_equity=197,  # 80+62+25+200+20 -30-150-10=197 to balance
        beginning_retained_earnings=100,
        interest_rate=0.05,
        dividends_pct_net_income=0.20,
        share_count=10,
        forecast_years=5,
    )


def test_model_builds_successfully():
    a = _base_assumptions()
    m = ThreeStatementModel(a)
    assert m.income_statement is not None
    assert m.balance_sheet is not None
    assert m.cash_flow is not None
    # Check dimensions
    assert m.income_statement.shape[1] == 5

def test_income_statement_revenue_progression():
    a = _base_assumptions()
    m = ThreeStatementModel(a)
    rev = m.income_statement.loc["Revenue"]
    # Each year revenue grows exactly by growth rate
    assert abs(rev.iloc[0] - 500 * 1.06) < 1e-6
    assert abs(rev.iloc[1] - rev.iloc[0] * 1.05) < 1e-6

def test_balance_sheet_identity():
    a = _base_assumptions()
    m = ThreeStatementModel(a)
    assert is_balanced(m)
    checks = run_all_checks(m)
    assert checks["balance_sheet"]["Pass"].all()

def test_cash_reconciliation():
    a = _base_assumptions()
    m = ThreeStatementModel(a)
    checks = run_all_checks(m)
    assert checks["cash"]["Cash Pass"].all()
    assert checks["cash"]["Recon Pass"].all()

def test_debt_reconciliation():
    a = _base_assumptions()
    m = ThreeStatementModel(a)
    checks = run_all_checks(m)
    assert checks["debt"]["Pass"].all()

def test_retained_earnings():
    a = _base_assumptions()
    m = ThreeStatementModel(a)
    checks = run_all_checks(m)
    assert checks["retained_earnings"]["Pass"].all()

def test_invalid_historical_revenue_raises():
    with pytest.raises(ValueError):
        ModelAssumptions(historical_revenue=-100, revenue_growth=[0.05] * 5, forecast_years=5)

def test_invalid_tax_rate_raises():
    with pytest.raises(ValueError):
        ModelAssumptions(historical_revenue=100, revenue_growth=[0.05] * 5, tax_rate=1.0, forecast_years=5)

def test_unbalanced_beginning_raises():
    with pytest.raises(ValueError):
        ModelAssumptions(
            historical_revenue=100,
            revenue_growth=[0.05] * 5,
            beginning_cash=100,
            beginning_ar=0,
            beginning_inventory=0,
            beginning_ppe=0,
            beginning_other_assets=0,
            beginning_ap=0,
            beginning_debt=0,
            beginning_other_liabilities=0,
            beginning_equity=50,  # should be 100
            forecast_years=5,
        )

def test_scenario_apply():
    base = _base_assumptions()
    bear = BEAR_SCENARIO.apply_to(base)
    # Bear has lower growth + lower margin
    assert bear.revenue_growth_list()[0] < base.revenue_growth_list()[0]
    assert bear.gross_margin_list()[0] < base.gross_margin_list()[0]

def test_scenario_comparison():
    base = _base_assumptions()
    df = scenario_comparison(
        base,
        scenarios=[BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO],
        wacc_map={"Bear": 0.10, "Base": 0.09, "Bull": 0.08},
        current_price=50,
    )
    assert df.shape[0] == 3
    assert "Implied Price (Growth)" in df.columns
    # Bull should price > Bear
    assert df.loc["Bull", "Implied Price (Growth)"] > df.loc["Bear", "Implied Price (Growth)"]

def test_negative_revenue_growth_below_minus_one_raises():
    a = _base_assumptions()
    a.revenue_growth = [-1.5, 0.05, 0.05, 0.05, 0.05]
    with pytest.raises(ValueError):
        ThreeStatementModel(a)

def test_forecast_labels():
    a = _base_assumptions()
    assert a.forecast_labels() == [2025, 2026, 2027, 2028, 2029]
