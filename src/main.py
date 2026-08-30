"""CLI entry point for the Financial Modeling & Quantitative Finance Toolkit.

Usage:
    python -m src.main                  # interactive menu
    python -m src.main --demo           # run full demo non-interactively
    python -m src.main three-statement  # run specific command
    python -m src.main dcf
    python -m src.main sensitivity
    python -m src.main scenarios
    python -m src.main portfolio-stats
    python -m src.main optimize
    python -m src.main frontier
    python -m src.main monte-carlo
    python -m src.main full-demo
    fmtk                                # if installed via pip

Each command prints results, runs checks, and writes artefacts to outputs/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure project root on path when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.financial_model.assumptions import ModelAssumptions, BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO
from src.financial_model.three_statement import ThreeStatementModel
from src.financial_model.checks import run_all_checks, is_balanced
from src.valuation.wacc import calculate_cost_of_equity, calculate_wacc
from src.valuation.dcf import DCFModel
from src.valuation.sensitivity import sensitivity_wacc_growth, sensitivity_wacc_exit_multiple, scenario_comparison
from src.portfolio.returns import simple_returns, annualized_return, annualized_volatility
from src.portfolio.risk import sharpe_ratio, max_drawdown, correlation_matrix, covariance_matrix
from src.portfolio.optimization import minimum_variance_portfolio, maximum_sharpe_portfolio, target_return_portfolio, monte_carlo_simulation
from src.portfolio.efficient_frontier import efficient_frontier


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

def _default_assumptions() -> ModelAssumptions:
    """Sensible demo assumptions that satisfy A=L+E."""
    return ModelAssumptions(
        historical_revenue=500,
        revenue_growth=[0.06, 0.05, 0.04, 0.03, 0.03],
        gross_margin=0.40,
        opex_margin=0.20,
        da_pct_revenue=0.03,
        capex_pct_revenue=0.04,
        tax_rate=0.25,
        ar_days=45, inventory_days=30, ap_days=40,
        beginning_cash=80, beginning_ar=62, beginning_inventory=25,
        beginning_ppe=200, beginning_other_assets=20,
        beginning_ap=30, beginning_debt=150, beginning_other_liabilities=10,
        beginning_equity=197, beginning_retained_earnings=100,
        interest_rate=0.05, dividends_pct_net_income=0.20, share_count=10,
        forecast_years=5, base_year_label=2024,
    )

def _default_prices(n_days: int = 1260) -> pd.DataFrame:
    """Synthetic but realistic price panel — labelled as synthetic."""
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "TLT"]
    mus = np.array([0.0006, 0.0005, 0.0003, 0.0002, 0.0001])
    vols = np.array([0.015, 0.013, 0.014, 0.012, 0.006])
    prices = pd.DataFrame(index=dates)
    for t, mu, vol in zip(tickers, mus, vols):
        prices[t] = 100 * np.cumprod(1 + np.random.normal(mu, vol, len(dates)))
    return prices

def _print_checks(model: ThreeStatementModel) -> None:
    checks = run_all_checks(model)
    # Structured block as in spec §7
    print("\nMODEL CHECKS")
    print("─" * 28)
    overall = True
    labels = {
        "balance_sheet": "Balance Sheet",
        "cash": "Cash Reconciliation",
        "debt": "Debt Reconciliation",
        "retained_earnings": "Equity Reconciliation",
    }
    for key, label in labels.items():
        df = checks[key]
        pass_cols = [c for c in df.columns if "Pass" in c]
        ok = all(df[c].all() for c in pass_cols) if pass_cols else False
        overall = overall and ok
        print(f"{label:<22} {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(df[~df[pass_cols].all(axis=1)] if pass_cols else df)
    print("─" * 28)
    print(f"{'Overall':<22} {'PASS' if overall else 'FAIL'}")
    print()


# --------------------------------------------------------------------------- #
# Command implementations
# --------------------------------------------------------------------------- #

def cmd_three_statement() -> ThreeStatementModel:
    print("\n== Three-Statement Model ==")
    assumptions = _default_assumptions()
    model = ThreeStatementModel(assumptions)
    print("\nIncome Statement:")
    print(model.income_statement.round(1).to_string())
    print("\nBalance Sheet:")
    print(model.balance_sheet.round(1).to_string())
    print("\nCash Flow Statement:")
    print(model.cash_flow.round(1).to_string())
    _print_checks(model)
    # Export
    model.to_excel(str(OUTPUTS / "three_statement.xlsx"))
    model.income_statement.to_csv(OUTPUTS / "income_statement.csv")
    model.balance_sheet.to_csv(OUTPUTS / "balance_sheet.csv")
    model.cash_flow.to_csv(OUTPUTS / "cash_flow.csv")
    print(f"Exported to {OUTPUTS / 'three_statement.xlsx'} (+ CSVs)")
    # Charts
    from src.visualization.charts import plot_revenue_forecast, plot_ebitda_margin
    plot_revenue_forecast(model.income_statement, save_path=str(OUTPUTS / "revenue_forecast.png"))
    plot_ebitda_margin(model.income_statement, save_path=str(OUTPUTS / "ebitda_margin.png"))
    print(f"Charts: {OUTPUTS / 'revenue_forecast.png'}, {OUTPUTS / 'ebitda_margin.png'}")
    return model


def cmd_dcf(model: ThreeStatementModel | None = None) -> DCFModel:
    print("\n== DCF Valuation ==")
    if model is None:
        model = ThreeStatementModel(_default_assumptions())
    ke = calculate_cost_of_equity(risk_free_rate=0.03, beta=1.1, equity_risk_premium=0.05)
    wacc = calculate_wacc(market_cap=800, debt=150, cost_of_equity=ke, cost_of_debt=0.05, tax_rate=0.25)
    print(f"Ke={ke:.2%}, WACC={wacc:.2%}  (Rf=3%, β=1.1, ERP=5%, Kd=5%, Tc=25%)")
    dcf = DCFModel.from_three_statement(model, wacc=wacc, terminal_growth=0.02, exit_multiple=10)
    sched = dcf.build()
    print("\nFCFF Schedule:")
    print(sched.round(1).to_string())
    for method in ("growth", "multiple"):
        s = dcf.summary(method)
        print(f"\n[{s['method']}] EV={s['enterprise_value']:.1f}  Equity={s['equity_value']:.1f}  Price={s['implied_share_price']:.2f}")
    # Export
    sched.to_csv(OUTPUTS / "dcf_schedule.csv")
    pd.DataFrame([dcf.summary("growth"), dcf.summary("multiple")]).to_csv(OUTPUTS / "dcf_results.csv", index=False)
    print(f"Exported: {OUTPUTS / 'dcf_results.csv'}, {OUTPUTS / 'dcf_schedule.csv'}")
    from src.visualization.charts import plot_fcff_forecast
    plot_fcff_forecast(sched, save_path=str(OUTPUTS / "fcff_forecast.png"))
    print(f"Chart: {OUTPUTS / 'fcff_forecast.png'}")
    return dcf


def cmd_sensitivity(dcf: DCFModel | None = None) -> None:
    print("\n== Sensitivity Analysis ==")
    if dcf is None:
        dcf = cmd_dcf()
    wacc_range = [0.07, 0.08, 0.09, 0.10, 0.11]
    growth_range = [0.01, 0.015, 0.02, 0.025, 0.03]
    mult_range = [8, 9, 10, 11, 12]
    sens_g = sensitivity_wacc_growth(dcf.inputs, wacc_range, growth_range)
    sens_m = sensitivity_wacc_exit_multiple(dcf.inputs, wacc_range, mult_range)
    print("\nWACC × Terminal Growth (Implied Price):")
    print(sens_g.round(1).to_string())
    print("\nWACC × Exit Multiple (Implied Price):")
    print(sens_m.round(1).to_string())
    sens_g.to_csv(OUTPUTS / "dcf_sensitivity_growth.csv")
    sens_m.to_csv(OUTPUTS / "dcf_sensitivity_multiple.csv")
    from src.visualization.charts import plot_sensitivity_heatmap
    plot_sensitivity_heatmap(sens_g, title="WACC × Terminal Growth → Implied Price", save_path=str(OUTPUTS / "dcf_sensitivity.png"), base_wacc="9.0%", base_growth="2.0%")
    plot_sensitivity_heatmap(sens_m, title="WACC × Exit Multiple → Implied Price", save_path=str(OUTPUTS / "dcf_sensitivity_multiple.png"))
    print(f"Exported: {OUTPUTS / 'dcf_sensitivity_growth.csv'}, {OUTPUTS / 'dcf_sensitivity_multiple.csv'}")
    print(f"Charts: {OUTPUTS / 'dcf_sensitivity.png'}, {OUTPUTS / 'dcf_sensitivity_multiple.png'}")


def cmd_scenarios() -> None:
    print("\n== Scenario Analysis (Bear / Base / Bull) ==")
    assumptions = _default_assumptions()
    df = scenario_comparison(
        assumptions,
        scenarios=[BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO],
        wacc_map={"Bear": 0.10, "Base": 0.09, "Bull": 0.08},
        terminal_growth_map={"Bear": 0.01, "Base": 0.02, "Bull": 0.025},
        current_price=80,
    )
    print(df.round(2).to_string())
    df.to_csv(OUTPUTS / "scenario_analysis.csv")
    print(f"Exported: {OUTPUTS / 'scenario_analysis.csv'}")


def cmd_portfolio_stats(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    print("\n== Portfolio Analytics ==")
    if prices is None:
        prices = _default_prices()
        print("Using synthetic demonstration dataset — not actual company financials.")
    rets = simple_returns(prices)
    ann_ret = annualized_return(rets)
    ann_vol = annualized_volatility(rets)
    sharpe = sharpe_ratio(rets, risk_free_rate=0.02)
    corr = correlation_matrix(rets)
    cov = covariance_matrix(rets)
    print("\nAnnualized Return / Vol / Sharpe (Rf=2%):")
    print(pd.DataFrame({"Return": ann_ret, "Vol": ann_vol, "Sharpe": sharpe}).round(4).to_string())
    print("\nCorrelation:")
    print(corr.round(2).to_string())
    print("\nCovariance (annualized):")
    print(cov.round(4).to_string())
    # Drawdown per name
    for col in prices.columns:
        dd = max_drawdown(prices=prices[col])
        print(f"Max DD {col}: {dd:.1%}")
    # Export
    pd.DataFrame({"Return": ann_ret, "Vol": ann_vol, "Sharpe": sharpe}).to_csv(OUTPUTS / "portfolio_statistics.csv")
    corr.to_csv(OUTPUTS / "correlation.csv")
    cov.to_csv(OUTPUTS / "covariance.csv")
    print(f"Exported: {OUTPUTS / 'portfolio_statistics.csv'} (+ corr/cov)")
    from src.visualization.charts import plot_cumulative_returns, plot_correlation_heatmap, plot_drawdown
    plot_cumulative_returns(prices=prices, save_path=str(OUTPUTS / "cumulative_returns.png"))
    plot_correlation_heatmap(corr, save_path=str(OUTPUTS / "correlation.png"))
    # drawdown of equal-weight
    eq_rets = rets.mean(axis=1)
    plot_drawdown(returns=eq_rets, title="Equal-Weight Drawdown", save_path=str(OUTPUTS / "drawdown.png"))
    print(f"Charts: {OUTPUTS / 'cumulative_returns.png'}, {OUTPUTS / 'correlation.png'}, {OUTPUTS / 'drawdown.png'}")
    return rets


def cmd_optimize(rets: pd.DataFrame | None = None) -> None:
    print("\n== Portfolio Optimization ==")
    if rets is None:
        rets = simple_returns(_default_prices())
    er = annualized_return(rets)
    cov = covariance_matrix(rets)
    min_var = minimum_variance_portfolio(er, cov, risk_free_rate=0.02)
    max_sh = maximum_sharpe_portfolio(er, cov, risk_free_rate=0.02)
    target = target_return_portfolio(er, cov, target_return=float(er.mean()), risk_free_rate=0.02)
    for name, res in [("Minimum Variance", min_var), ("Maximum Sharpe", max_sh), ("Target Return (≈mean)", target)]:
        print(f"\n{name}: Ret={res['expected_return']:.2%} Vol={res['volatility']:.2%} Sharpe={res['sharpe_ratio']:.2f}")
        print(res["weights"].round(4).to_string())
    pd.DataFrame({
        "MinVar": min_var["weights"],
        "MaxSharpe": max_sh["weights"],
        "Target": target["weights"],
    }).to_csv(OUTPUTS / "optimal_portfolios.csv")
    # Also save metrics
    pd.DataFrame([
        {"portfolio": "MinVar", **{k: v for k, v in min_var.items() if k != "weights"}},
        {"portfolio": "MaxSharpe", **{k: v for k, v in max_sh.items() if k != "weights"}},
        {"portfolio": "Target", **{k: v for k, v in target.items() if k != "weights"}},
    ]).to_csv(OUTPUTS / "optimal_portfolio_metrics.csv", index=False)
    print(f"Exported: {OUTPUTS / 'optimal_portfolios.csv'}, {OUTPUTS / 'optimal_portfolio_metrics.csv'}")


def cmd_frontier(rets: pd.DataFrame | None = None) -> None:
    print("\n== Efficient Frontier ==")
    if rets is None:
        rets = simple_returns(_default_prices())
    er = annualized_return(rets)
    cov = covariance_matrix(rets)
    front = efficient_frontier(er, cov, n_points=30, risk_free_rate=0.02)
    print(front.head().round(4).to_string())
    front.to_csv(OUTPUTS / "efficient_frontier.csv", index=False)
    min_var = minimum_variance_portfolio(er, cov, risk_free_rate=0.02)
    max_sh = maximum_sharpe_portfolio(er, cov, risk_free_rate=0.02)
    mc = monte_carlo_simulation(er, cov, n_portfolios=3000, seed=42, risk_free_rate=0.02)
    from src.visualization.charts import plot_efficient_frontier
    plot_efficient_frontier(front, min_var=min_var, max_sharpe=max_sh, mc_df=mc, save_path=str(OUTPUTS / "efficient_frontier.png"))
    print(f"Exported: {OUTPUTS / 'efficient_frontier.csv'}")
    print(f"Chart: {OUTPUTS / 'efficient_frontier.png'}")


def cmd_monte_carlo(rets: pd.DataFrame | None = None) -> None:
    print("\n== Monte Carlo Simulation ==")
    if rets is None:
        rets = simple_returns(_default_prices())
    er = annualized_return(rets)
    cov = covariance_matrix(rets)
    mc = monte_carlo_simulation(er, cov, n_portfolios=5000, seed=42, risk_free_rate=0.02)
    print(mc.describe().round(4).to_string())
    best = mc.loc[mc["sharpe"].idxmax()]
    print(f"\nBest Sharpe portfolio in simulation: Ret={best['return']:.2%} Vol={best['volatility']:.2%} Sharpe={best['sharpe']:.2f}")
    mc.to_csv(OUTPUTS / "monte_carlo_simulation.csv", index=False)
    # Scatter
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(mc["volatility"], mc["return"], c=mc["sharpe"], cmap="viridis", alpha=0.45, s=10)
    ax.scatter([best["volatility"]], [best["return"]], color="red", s=120, marker="*", edgecolors="black", label="Best Sharpe (sim)")
    ax.set_xlabel("Volatility"); ax.set_ylabel("Expected Return")
    ax.set_title("Monte Carlo Risk-Return (Sharpe color)")
    plt.colorbar(sc, label="Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "portfolio_simulation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Exported: {OUTPUTS / 'monte_carlo_simulation.csv'}")
    print(f"Chart: {OUTPUTS / 'portfolio_simulation.png'}")


def cmd_full_demo() -> None:
    print("\n" + "="*60)
    print(" FULL DEMONSTRATION — Financial Modeling & Quant Toolkit")
    print(" Synthethic demonstration datasets — not actual company financials.")
    print("="*60)
    model = cmd_three_statement()
    dcf = cmd_dcf(model)
    cmd_sensitivity(dcf)
    cmd_scenarios()
    rets = cmd_portfolio_stats()
    cmd_optimize(rets)
    cmd_frontier(rets)
    cmd_monte_carlo(rets)
    print("\n" + "="*60)
    print(" DEMO COMPLETE — all artefacts in outputs/")
    print("  dcf_results.csv, dcf_schedule.csv, dcf_sensitivity.png,")
    print("  scenario_analysis.csv, portfolio_statistics.csv,")
    print("  optimal_portfolios.csv, efficient_frontier.png,")
    print("  portfolio_simulation.png, drawdown.png, + more")
    print("="*60 + "\n")


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #

COMMANDS = {
    "three-statement": lambda _: cmd_three_statement(),
    "dcf": lambda _: cmd_dcf(),
    "sensitivity": lambda _: cmd_sensitivity(),
    "scenarios": lambda _: cmd_scenarios(),
    "portfolio-stats": lambda _: cmd_portfolio_stats(),
    "optimize": lambda _: cmd_optimize(),
    "frontier": lambda _: cmd_frontier(),
    "monte-carlo": lambda _: cmd_monte_carlo(),
    "full-demo": lambda _: cmd_full_demo(),
    "demo": lambda _: cmd_full_demo(),
}

def _interactive_menu() -> None:
    title = """
Financial Modeling & Quantitative Finance Toolkit

1. Run Three-Statement Model
2. Run DCF Valuation
3. Run Sensitivity Analysis
4. Run Scenario Analysis
5. Run Portfolio Analytics
6. Run Portfolio Optimization
7. Generate Efficient Frontier
8. Run Monte Carlo Simulation
9. Run Full Demonstration
0. Exit
"""
    actions = {
        "1": cmd_three_statement,
        "2": cmd_dcf,
        "3": cmd_sensitivity,
        "4": cmd_scenarios,
        "5": cmd_portfolio_stats,
        "6": cmd_optimize,
        "7": cmd_frontier,
        "8": cmd_monte_carlo,
        "9": cmd_full_demo,
    }
    while True:
        print(title)
        choice = input("Select [0-9]: ").strip()
        if choice == "0":
            print("Bye.")
            break
        if choice in actions:
            try:
                actions[choice]()
            except Exception as e:
                print(f"\n[ERROR] {e}\n", file=sys.stderr)
        else:
            print("Invalid choice. Try 0-9.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Financial Modeling & Quantitative Finance Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python -m src.main --demo\n  python -m src.main full-demo\n  python -m src.main dcf\n  fmtk --help\n",
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS.keys()), help="Command to run (omit for interactive menu)")
    parser.add_argument("--demo", action="store_true", help="Run full demonstration (alias for full-demo)")
    parser.add_argument("--list", action="store_true", help="List available commands and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("Available commands:")
        for k in sorted(COMMANDS):
            print(f"  {k}")
        return 0

    if args.demo or args.command in ("full-demo", "demo"):
        cmd_full_demo()
        return 0

    if args.command:
        COMMANDS[args.command](None)
        return 0

    # No command → interactive menu (only if TTY; otherwise run demo)
    if sys.stdin.isatty():
        _interactive_menu()
    else:
        print("No command given and not a TTY — running full demo.")
        cmd_full_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
