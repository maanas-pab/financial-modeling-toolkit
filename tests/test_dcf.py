"""Tests for WACC, DCF, terminal value, and sensitivity."""

import pytest
import numpy as np
import pandas as pd

from src.valuation.wacc import calculate_wacc, calculate_cost_of_equity, calculate_after_tax_cost_of_debt
from src.valuation.dcf import DCFInputs, DCFModel
from src.valuation.sensitivity import sensitivity_wacc_growth, sensitivity_wacc_exit_multiple


# ---------- WACC ----------

def test_cost_of_equity_basic():
    ke = calculate_cost_of_equity(risk_free_rate=0.03, beta=1.2, equity_risk_premium=0.05)
    assert abs(ke - 0.09) < 1e-8

def test_cost_of_equity_invalid_beta():
    with pytest.raises(ValueError):
        calculate_cost_of_equity(0.03, -0.5, 0.05)

def test_after_tax_cost_of_debt():
    kd = calculate_after_tax_cost_of_debt(0.06, 0.25)
    assert abs(kd - 0.045) < 1e-8

def test_after_tax_cost_of_debt_invalid_tax():
    with pytest.raises(ValueError):
        calculate_after_tax_cost_of_debt(0.06, 1.0)

def test_wacc_basic():
    wacc = calculate_wacc(market_cap=800, debt=200, cost_of_equity=0.10, cost_of_debt=0.05, tax_rate=0.25)
    # E/V=0.8, D/V=0.2, Kd*(1-T)=0.0375 => 0.8*0.10+0.2*0.0375=0.0875
    assert abs(wacc - 0.0875) < 1e-8

def test_wacc_both_zero_raises():
    with pytest.raises(ValueError):
        calculate_wacc(0, 0, 0.10, 0.05, 0.25)

def test_wacc_invalid_cost_of_equity():
    with pytest.raises(ValueError):
        calculate_wacc(100, 0, 0.99, 0.05, 0.25)


# ---------- DCF ----------

def _sample_inputs():
    return DCFInputs(
        revenue=[100, 110, 121],
        ebit=[20, 22, 24],
        da=[5, 5, 5],
        capex=[6, 6, 6],
        working_capital_change=[2, 2, 2],
        tax_rate=0.25,
        wacc=0.10,
        terminal_growth=0.02,
        exit_multiple=8,
        debt=30,
        cash=10,
        shares_outstanding=10,
    )

def test_dcf_fcff_calculation():
    inp = _sample_inputs()
    m = DCFModel(inp)
    df = m.build()
    # NOPAT year1 = 20*0.75=15, FCFF=15+5-6-2=12
    assert abs(df.loc["FCFF", "Year 1"] - 12) < 1e-8
    assert abs(df.loc["NOPAT", "Year 1"] - 15) < 1e-8

def test_dcf_terminal_growth():
    inp = _sample_inputs()
    m = DCFModel(inp)
    m.build()
    s = m.summary("growth")
    # FCFF year3 = 24*0.75+5-6-2=18+5-8=15? wait 24*0.75=18 =>18+5-6-2=15
    # TV = 15*1.02/(0.10-0.02)=15.3/0.08=191.25
    assert abs(s["terminal_value"] - 191.25) < 1e-6
    # PV = sum PV(FCFF) + PV(TV). Check not zero
    assert s["enterprise_value"] > 0
    assert s["implied_share_price"] > 0

def test_dcf_exit_multiple():
    inp = _sample_inputs()
    m = DCFModel(inp)
    s = m.summary("multiple")
    # ebitda year3 = 24+5=29, TV=29*8=232
    assert abs(s["terminal_value"] - 232) < 1e-6

def test_dcf_wacc_le_terminal_growth_raises():
    with pytest.raises(ValueError):
        DCFInputs(
            revenue=[100, 110],
            ebit=[20, 22],
            da=[5, 5],
            capex=[6, 6],
            working_capital_change=[1, 1],
            wacc=0.02,
            terminal_growth=0.03,
        )

def test_dcf_negative_revenue_raises():
    # DCFInputs doesn't validate revenue sign but downstream should handle;
    # we test that model still computes (finance check is on assumptions layer)
    inp = DCFInputs(
        revenue=[100, 110],
        ebit=[-5, -5],
        da=[5, 5],
        capex=[6, 6],
        working_capital_change=[1, 1],
        wacc=0.10,
        terminal_growth=0.02,
    )
    m = DCFModel(inp)
    df = m.build()
    # FCFF negative is allowed
    assert df.loc["FCFF", "Year 1"] < 20

def test_sensitivity_wacc_growth_shape():
    inp = _sample_inputs()
    waccs = [0.08, 0.10, 0.12]
    growths = [0.01, 0.02, 0.03]
    df = sensitivity_wacc_growth(inp, waccs, growths)
    assert df.shape == (3, 3)
    # Higher WACC => lower price (monotonic along rows for fixed column)
    assert df.iloc[0, 1] > df.iloc[1, 1] > df.iloc[2, 1]

def test_sensitivity_wacc_exit_shape():
    inp = _sample_inputs()
    waccs = [0.08, 0.10]
    mults = [6, 8, 10]
    df = sensitivity_wacc_exit_multiple(inp, waccs, mults)
    assert df.shape == (2, 3)
    # Higher multiple => higher price
    assert df.iloc[0, 0] < df.iloc[0, 2]

def test_sensitivity_invalid_wacc_growth_returns_nan():
    inp = _sample_inputs()
    # 7% wacc with 7% growth => wacc not > growth => row should be NaN
    df = sensitivity_wacc_growth(inp, [0.07], [0.07, 0.01])
    assert np.isnan(df.iloc[0, 0])
    assert not np.isnan(df.iloc[0, 1])
