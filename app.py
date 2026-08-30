"""Streamlit dashboard for Financial Modeling Toolkit — interactive window.

Run: streamlit run app.py
or: python -m streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.financial_model.assumptions import ModelAssumptions
from src.financial_model.three_statement import ThreeStatementModel
from src.financial_model.checks import run_all_checks
from src.valuation.wacc import calculate_cost_of_equity, calculate_wacc
from src.valuation.dcf import DCFModel
from src.portfolio.returns import simple_returns
from src.portfolio.risk import covariance_matrix
from src.portfolio.optimization import minimum_variance_portfolio, maximum_sharpe_portfolio

st.set_page_config(page_title="Financial Modeling Toolkit", layout="wide")

st.title("Financial Modeling & Quantitative Finance Toolkit")
st.caption("Synthetic demo — not actual company financials. All logic in `src/`; this is just the window.")

tab1, tab2, tab3 = st.tabs(["🏦 Three-Statement & DCF", "📊 Sensitivity & Scenarios", "💼 Portfolio"])

with tab1:
    st.header("Three-Statement + DCF")
    c1, c2, c3, c4 = st.columns(4)
    hist_rev = c1.number_input("Historical Revenue", 100.0, 5000.0, 500.0, 10.0)
    g1 = c2.slider("Growth Y1", 0.0, 0.20, 0.06, 0.01)
    g2 = c3.slider("Growth Y2-5 avg", 0.0, 0.20, 0.04, 0.01)
    tax = c4.slider("Tax rate", 0.0, 0.5, 0.25, 0.01)
    c5, c6, c7, c8 = st.columns(4)
    gm = c5.slider("Gross margin", 0.1, 0.7, 0.40, 0.01)
    om = c6.slider("Opex margin", 0.05, 0.4, 0.20, 0.01)
    da_pct = c7.slider("D&A % Rev", 0.0, 0.1, 0.03, 0.005)
    capex_pct = c8.slider("CapEx % Rev", 0.0, 0.15, 0.04, 0.005)

    st.subheader("WACC")
    wc1, wc2, wc3, wc4 = st.columns(4)
    rf = wc1.number_input("Rf", 0.0, 0.1, 0.03, 0.005, format="%.3f")
    beta = wc2.number_input("Beta", 0.5, 3.0, 1.1, 0.1)
    erp = wc3.number_input("ERP", 0.01, 0.1, 0.05, 0.005, format="%.3f")
    kd = wc4.number_input("Cost of debt", 0.01, 0.15, 0.05, 0.005, format="%.3f")

    if st.button("Run Model", type="primary"):
        assump = ModelAssumptions(
            historical_revenue=hist_rev,
            revenue_growth=[g1, g2, g2, g2, g2],
            gross_margin=gm, opex_margin=om, da_pct_revenue=da_pct, capex_pct_revenue=capex_pct,
            tax_rate=tax, beginning_cash=80, beginning_ar=62, beginning_inventory=25,
            beginning_ppe=200, beginning_other_assets=20, beginning_ap=30, beginning_debt=150,
            beginning_other_liabilities=10, beginning_equity=197, beginning_retained_earnings=100,
            forecast_years=5
        )
        model = ThreeStatementModel(assump)
        st.dataframe(model.income_statement.round(1), use_container_width=True)
        st.dataframe(model.balance_sheet.round(1), use_container_width=True)
        checks = run_all_checks(model)
        for k, df in checks.items():
            st.write(k, df)

        ke = calculate_cost_of_equity(rf, beta, erp)
        wacc = calculate_wacc(800, 150, ke, kd, tax)
        dcf = DCFModel.from_three_statement(model, wacc=wacc, terminal_growth=0.02, exit_multiple=10)
        dcf.build()
        cA, cB = st.columns(2)
        cA.metric("WACC", f"{wacc:.2%}"); cA.metric("Ke", f"{ke:.2%}")
        cB.json(dcf.summary("growth"))
        cB.json(dcf.summary("multiple"))
        st.dataframe(dcf.to_dataframe().round(1), use_container_width=True)

with tab2:
    st.header("Sensitivity & Scenarios (run from Tab 1 first or use defaults)")
    st.info("For full grids run: `python -m src.main sensitivity` or `scenarios` — demo below uses defaults.")
    from src.valuation.sensitivity import sensitivity_wacc_growth, scenario_comparison
    from src.financial_model.assumptions import BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO
    if st.button("Run Sensitivity & Scenarios"):
        assump = ModelAssumptions(historical_revenue=500, revenue_growth=[0.06,0.05,0.04,0.03,0.03],
            gross_margin=0.40, opex_margin=0.20, da_pct_revenue=0.03, capex_pct_revenue=0.04, tax_rate=0.25,
            beginning_cash=80, beginning_ar=62, beginning_inventory=25, beginning_ppe=200,
            beginning_other_assets=20, beginning_ap=30, beginning_debt=150, beginning_other_liabilities=10,
            beginning_equity=197, beginning_retained_earnings=100, forecast_years=5)
        m = ThreeStatementModel(assump)
        dcf = DCFModel.from_three_statement(m, wacc=0.09, terminal_growth=0.02, exit_multiple=10)
        sens = sensitivity_wacc_growth(dcf.inputs, [0.08,0.09,0.10], [0.01,0.02,0.03])
        st.dataframe(sens.round(1), use_container_width=True)
        fig, ax = plt.subplots()
        im = ax.imshow(sens.values, cmap="RdYlGn")
        ax.set_xticks(range(len(sens.columns))); ax.set_xticklabels(sens.columns)
        ax.set_yticks(range(len(sens.index))); ax.set_yticklabels(sens.index)
        plt.colorbar(im, ax=ax, label="Price")
        st.pyplot(fig)

        from src.valuation.sensitivity import scenario_comparison
        scen = scenario_comparison(assump, [BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO],
            wacc_map={"Bear":0.10,"Base":0.09,"Bull":0.08}, current_price=80)
        st.dataframe(scen.round(2), use_container_width=True)

with tab3:
    st.header("Portfolio — Synthetic (seed 42)")
    if st.button("Generate Portfolio"):
        np.random.seed(42)
        dates = pd.date_range("2018-01-01", periods=1260, freq="B")
        tickers = ["AAPL","MSFT","JPM","XOM","TLT"]
        mus = [0.0006,0.0005,0.0003,0.0002,0.0001]
        vols = [0.015,0.013,0.014,0.012,0.006]
        prices = pd.DataFrame(index=dates)
        for t, mu, vol in zip(tickers, mus, vols):
            prices[t] = 100 * np.cumprod(1 + np.random.normal(mu, vol, len(dates)))
        rets = simple_returns(prices)
        st.line_chart((1+rets).cumprod())
        ann_ret = (1+rets).prod()**(252/len(rets))-1
        st.write(ann_ret)
        cov = covariance_matrix(rets)
        st.dataframe(cov.round(4))
        try:
            min_var = minimum_variance_portfolio(ann_ret, cov)
            max_sh = maximum_sharpe_portfolio(ann_ret, cov, risk_free_rate=0.02)
            st.json({k: float(v) for k,v in min_var.items() if k!="weights"})
            st.write("Min Var weights", min_var["weights"].round(3).to_dict())
            st.write("Max Sharpe weights", max_sh["weights"].round(3).to_dict())
        except Exception as e:
            st.error(str(e))

st.sidebar.markdown("**CLI:** `python -m src.main`  \n**Demo:** `python examples/run_full_demo.py`  \n**Tests:** `pytest -v`")
