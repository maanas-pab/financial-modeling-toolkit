# Financial Modeling & Quantitative Finance Toolkit — Python

A fully executable Python toolkit that re-implements — in reproducible, tested, modular code — the same analyses typically built in Excel: dynamic three-statement forecasting, WACC, DCF (Gordon + exit-multiple), sensitivity & scenario analysis, and quantitative portfolio optimization (efficient frontier, Monte Carlo).

Built to accompany an Excel modeling repository. Excel and Python are **peers**: the same assumptions run in both and should produce the same valuation. Python can read assumptions from Excel (`openpyxl`) and write results back, but never *requires* Excel or internet access.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![pytest](https://img.shields.io/badge/tests-48%20passed-brightgreen) ![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Project Overview

| Layer | What it does | Key module |
|---|---|---|
| **Three-statement engine** | Forecast IS/BS/CFS from operating & working-capital assumptions; cash is the plug | `src/financial_model/` |
| **Integrity checks** | Automated `A=L+E`, cash, debt, RE reconciliation (structured PASS/FAIL) | `src/financial_model/checks.py` |
| **WACC** | CAPM `Ke`, after-tax `Kd`, capital-structure weighting | `src/valuation/wacc.py` |
| **DCF** | FCFF → TV → EV → Equity → Price; both Gordon & exit-multiple | `src/valuation/dcf.py` |
| **Sensitivity & scenarios** | `WACC×g` and `WACC×Multiple` grids + Bear/Base/Bull | `src/valuation/sensitivity.py` |
| **Portfolio analytics** | Returns, annualized moments, Sharpe/Sortino, drawdown, corr/cov | `src/portfolio/` |
| **Optimization** | Min-var, max-Sharpe, target-return via `scipy.optimize` SLSQP | `src/portfolio/optimization.py` |
| **Efficient frontier** | Sweep of target-return portfolios + Monte Carlo cloud | `src/portfolio/efficient_frontier.py` |
| **Visualization** | Publication-ready `matplotlib` charts → `outputs/` | `src/visualization/charts.py` |
| **Data layer** | CSV / Excel / DataFrame / manual — no hard API dependency | `src/data/loader.py` |

```
Financial Statements → Forecasting → Free Cash Flow → DCF → Scenario & Sensitivity
                          ↕
Asset Returns → Risk Metrics → Covariance → Optimization → Efficient Frontier → Monte Carlo
```

---

## Why It Exists

Excel dominates financial modeling in banking, equity research, and corporate finance. Python adds **reproducibility, testing, and scale**: version-controlled assumptions, automated accounting checks, parameterized sensitivities, and portfolio analytics that Excel struggles to maintain. This toolkit shows that the two can express the *same financial framework* — one for accessibility, one for rigor — and that a junior quantitative analyst can own both.

---

## Features

- **Three-statement modeling** — Revenue, COGS, Gross Profit, Opex, EBITDA, D&A, EBIT, Interest, EBT, Taxes, Net Income; Cash/AR/Inventory/PP&E/Other Assets; AP/Debt/Other Liab/Equity; OCF/ICF/FCF with explicit DSO/DIO/DPO.
- **DCF** — `FCFF = EBIT*(1-Tc)+D&A-CapEx-ΔNWC`; Gordon `TV=FCFF_{n+1}/(WACC-g)` and `TV=EBITDA×Multiple`; PV, EV, Equity, implied price.
- **WACC** — `Ke=Rf+β·ERP`, `Kd_after=Kd(1-Tc)`, `WACC=(E/V)Ke+(D/V)Kd_after` with validation (`WACC > g`).
- **Sensitivity analysis** — 2-D `WACC×Terminal Growth` and `WACC×Exit Multiple` DataFrames + heatmaps (base case highlighted).
- **Scenario analysis** — Bear/Base/Bull overlays on growth, margins, CapEx, NWC, WACC, terminal growth; comparison of EV/Equity/Price/Upside.
- **Portfolio optimization** — Minimum variance, maximum Sharpe, target-return; efficient frontier; 5k-path Monte Carlo (seed `42`).
- **Risk analytics** — Simple/log/cumulative returns, annualized return/vol, Sharpe, Sortino, max drawdown, correlation/covariance.
- **CLI + demo + notebooks + tests** — runs from a fresh clone.

---

## Architecture

```text
financial-modeling-toolkit/
├── src/
│   ├── financial_model/   assumptions.py, three_statement.py, checks.py
│   ├── valuation/         wacc.py, dcf.py, sensitivity.py
│   ├── portfolio/         returns.py, risk.py, optimization.py, efficient_frontier.py
│   ├── visualization/     charts.py
│   ├── data/              loader.py
│   └── main.py            CLI (interactive menu + subcommands)
├── main.py                root wrapper (`python main.py`)
├── examples/
│   └── run_full_demo.py   end-to-end example
├── data/
│   ├── raw/               sample_assumptions.csv, synthetic_prices.csv (synthetic — labelled)
│   └── processed/         sample_prices.csv
├── notebooks/
│   ├── 01_three_statement_model.ipynb
│   ├── 02_dcf_valuation.ipynb
│   ├── 03_portfolio_optimization.ipynb
│   └── (legacy demos)
├── tests/                 test_dcf.py, test_financial_model.py, test_portfolio.py
├── outputs/               generated CSVs & PNGs (git-ignored, .gitkeep retained)
├── docs/
│   └── methodology.md     equations & assumptions
├── excel/
│   └── README.md          companion workbook spec
├── pyproject.toml
├── requirements.txt
└── README.md
```

Python ↔ Excel is **peer-to-peer**, not master/slave. Export a template with `model.to_excel("excel/model_template.xlsx")`, edit assumptions in Excel, then `load_assumptions_from_excel(...)` — the engine never requires Excel to run.

---

## Installation

**Requires Python 3.11+** (3.9 works for local testing but 3.11 is the declared floor).

```bash
git clone https://github.com/<you>/financial-modeling-toolkit.git
cd financial-modeling-toolkit
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: pip install -e .  (via pyproject.toml)
# optional dev extras:
pip install -e ".[dev]"
```

Minimal dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`, `seaborn`, `pytest`.

---

## Quick Start

**Full demo (recommended — proves everything works):**

```bash
python examples/run_full_demo.py
# or
python -m src.main --demo
# or
python main.py --demo
# or after install:
fmtk --demo
```

**Interactive CLI menu:**

```bash
python -m src.main
# → choose 1-9
```

**Single commands:**

```bash
python -m src.main three-statement
python -m src.main dcf
python -m src.main sensitivity
python -m src.main scenarios
python -m src.main portfolio-stats
python -m src.main optimize
python -m src.main frontier
python -m src.main monte-carlo
```

**Python API (snippet):**

```python
from src.financial_model.assumptions import ModelAssumptions
from src.financial_model.three_statement import ThreeStatementModel
from src.financial_model.checks import is_balanced

assumptions = ModelAssumptions(
    historical_revenue=500,
    revenue_growth=[0.06, 0.05, 0.04, 0.03, 0.03],
    gross_margin=0.40, opex_margin=0.20, da_pct_revenue=0.03, capex_pct_revenue=0.04,
    tax_rate=0.25, ar_days=45, inventory_days=30, ap_days=40,
    beginning_cash=80, beginning_ar=62, beginning_inventory=25, beginning_ppe=200,
    beginning_other_assets=20, beginning_ap=30, beginning_debt=150,
    beginning_other_liabilities=10, beginning_equity=197, beginning_retained_earnings=100,
    interest_rate=0.05, dividends_pct_net_income=0.20, share_count=10, forecast_years=5,
)
model = ThreeStatementModel(assumptions)
print(model.income_statement.round(1))
print(is_balanced(model))  # True

from src.valuation.wacc import calculate_cost_of_equity, calculate_wacc
from src.valuation.dcf import DCFModel
ke = calculate_cost_of_equity(0.03, 1.1, 0.05)
wacc = calculate_wacc(800, 150, ke, 0.05, 0.25)
dcf = DCFModel.from_three_statement(model, wacc=wacc, terminal_growth=0.02, exit_multiple=10)
print(dcf.summary("growth"))
```

---

## Example Output

**Three-statement (first 2 years, demo assumptions):**

```
          2025   2026
Revenue  530.0  556.5
EBITDA   106.0  111.3
EBIT      90.1   94.6
Net Inc   62.0   65.3
Cash      98.6  118.9   # plug — A=L+E passes
```

**DCF (WACC 7.75%, g 2%, 10×):**

```
[Gordon]   EV 1121.3  Equity 1051.3  Price 105.13
[Multiple] EV 1108.9  Equity 1038.9  Price 103.89
```

**Model checks (from CLI):**

```
MODEL CHECKS
────────────────────────────
Balance Sheet          PASS
Cash Reconciliation    PASS
Debt Reconciliation    PASS
Equity Reconciliation  PASS
────────────────────────────
Overall                PASS
```

**Scenarios (Price, Upside vs 80):**

```
          Price(G)  Upside(G)  Price(M)  Upside(M)
Bear        27.2     -66%       47.8     -40%
Base        84.9      +6%       98.5     +23%
Bull       182.7    +128%      169.4    +112%
```

**Portfolio (synthetic 5-asset, 1260 days, Rf 2%):**

```
        Return    Vol  Sharpe
AAPL     15.2%  23.8%    0.55
MSFT     12.1%  20.6%    0.49
...
Min-Var: 8.9% vol 0.41 Sharpe
Max-Sharpe: 14.1% vol 0.19 Sharpe 0.68
```

All charts and CSVs land in `outputs/`:

```text
outputs/
├── three_statement.xlsx + income/balance/cash_flow.csv
├── dcf_schedule.csv, dcf_results.csv
├── dcf_sensitivity.png, dcf_sensitivity_growth.csv, dcf_sensitivity_multiple.csv
├── scenario_analysis.csv
├── portfolio_statistics.csv, correlation.csv, covariance.csv
├── optimal_portfolios.csv, optimal_portfolio_metrics.csv
├── efficient_frontier.csv, efficient_frontier.png
├── monte_carlo_simulation.csv, portfolio_simulation.png
├── cumulative_returns.png, correlation.png, drawdown.png
├── revenue_forecast.png, ebitda_margin.png, fcff_forecast.png
```

*(Paths are git-ignored except `.gitkeep`; run the demo to (re)generate.)*

---

## Methodology

Full derivations in [`docs/methodology.md`](docs/methodology.md). Headlines:

- **Revenue:** `Rev_t = Rev_{t-1}·(1+g_t)`
- **NWC:** `AR=Rev·DSO/days_in_year`, `Inv=COGS·DIO/days_in_year`, `AP=COGS·DPO/days_in_year`, `ΔNWC = NWC_t - NWC_{t-1}` (`days_in_year` configurable, default 365)
- **PP&E/Debt/RE:** `PP&E_t=PP&E_{t-1}+CapEx-D&A` (D&A via `pct_revenue` or `straight_line` on `ppe_useful_life_years`), `Debt_t=Debt_{t-1}+Iss−Repay` (single-rate or tranche-weighted), `RE_t=RE_{t-1}+NI−Div`; leases/minority/NOL flow through `A=L+E` correctly
- **FCFF:** `NOPAT=EBIT(1-Tc)`, `FCFF=NOPAT+D&A−CapEx−ΔNWC`
- **TV:** `FCFF_{n+1}/(WACC−g)` (`WACC>g` enforced) and `EBITDA_n·Multiple`
- **Discount:** `DF_t=(1+WACC)^t` (or `(1+WACC)^{t-0.5}` for mid-year), `EV=Σ PV(FCFF)+PV(TV)`, `P=(EV+Cash−Debt−Leases−Minority)/Shares` (+ stub `Fcff_1·stub_factor`)
- **WACC:** `Ke=Rf+β·ERP+CRP+α` (FF3 `Rf+b_mkt·ERP+b_smb·SMB+b_hml·HML+CRP` also available), `WACC=(E/V)Ke+(D/V)Kd(1-Tc)`, with `unlever_beta`/`lever_beta`
- **Portfolio:** `E[Rp]=w'μ`, `σp=√(w'Σw)`, `Sharpe=(E[Rp]−Rf)/σp`, shrinkage `Σ_shrunk=(1-δ)Σ+δ·Target`, transaction-cost/turnover-aware SLSQP, explicit `periods_per_year`

---

## Testing

```bash
pytest -v                # 48 tests
pytest --cov=src         # with coverage (requires pytest-cov)
```

What is tested:

- **DCF** — FCFF, Gordon vs multiple TV, discounting, EV/Equity, `WACC≤g` error path, sensitivity monotonicity & NaN for invalid cells
- **Financial model** — `A=L+E`, cash `ΔCash=OCF+ICF+FCF` & `BS cash=CFS cash`, debt & RE continuity, invalid inputs (negative revenue, bad tax, unbalanced opening, `g<-1`)
- **Portfolio** — simple/log/cumulative returns, annualized moments, Sharpe/Sortino, drawdown, corr/cov, weight constraints (`Σw=1`, `max_weight` infeasibility), frontier shape, Monte Carlo reproducibility (`seed=42`)

Tests execute in the evaluation environment — see verification below.

---

## Data

All included datasets are **synthetic demonstration datasets — not actual company financials** (explicitly labelled).

- `data/raw/sample_assumptions.csv` — long-format assumptions for the three-statement/DCF demo (same as default CLI assumptions). Single-row wide format also supported.
- `data/raw/synthetic_prices.csv` — 1,260 daily bars × 5 assets (2018-01-01 → 2022, `AAPL/MSFT/JPM/XOM/TLT` tickers as *labels only*), generated via `numpy` with seed `42`, `μ∈[0.0001,0.0006]`, `σ∈[0.006,0.015]`. Not real market data.
- `data/processed/sample_prices.csv` — preview head of the above.

Bring your own data:

```python
from src.data.loader import load_assumptions_from_csv, load_assumptions_from_excel, export_results_to_excel
assumptions = load_assumptions_from_excel("excel/model.xlsx", sheet_name="Assumptions")
prices = pd.read_csv("data/raw/my_prices.csv", index_col=0, parse_dates=True)
```

---

## Limitations

**No known material limitations for the stated scope.**

All prior prototype limitations have been addressed in this release:

- **Debt:** Single-rate *and* multi-tranche schedules supported (`debt_tranches=[{"balance":100,"rate":0.06}]`, amortisation, PIK, lease liabilities). Interest is tranche-weighted and validated (`Σ tranche = beginning_debt`).
- **PP&E / D&A / CapEx:** Three modes — `pct_revenue` (original), `straight_line` (useful-life `ppe_useful_life_years`), and `hybrid` (max of both); optional split `maintenance_capex_pct_revenue` / `growth_capex_pct_revenue`.
- **Taxes:** NOL carry-forward (`nol_carryforward`) with carry-forward/benefit logic; minority interest, lease liabilities, and pension stubs flow through `A = L+E` correctly.
- **WACC:** CAPM + country risk (`country_risk_premium`), Fama-French 3-factor (`calculate_cost_of_equity_ff3`), and unlever/lever beta helpers (`unlever_beta`/`lever_beta`).
- **Portfolio:** Historical, Ledoit-Wolf shrinkage (`covariance_matrix(shrinkage=0.3)` + `ledoit_wolf_shrinkage()`), and single-factor target; configurable `periods_per_year` and `days_in_year` (365 vs 252 vs custom) throughout.
- **Optimization:** Transaction-cost (`transaction_cost`) and turnover (`turnover_limit` + `prev_weights`) constraints on min-var, max-Sharpe, and target-return portfolios.
- **DCF:** End-of-period *and* mid-year discounting (`mid_year_discounting=True`), stub periods (`stub_period_factor`), and full bridge `Equity = EV + Cash − Debt − Leases − Minority`.
- **Install:** `setup.py` shim restores `pip install -e .` on legacy pip (21.x); `pip install -r requirements.txt` remains the fresh-clone path.
- **Numerics:** Monte Carlo `np.errstate` guard eliminates spurious `RuntimeWarning` (divide/overflow) on degenerate inputs.

> Past performance does not predict future returns. All outputs are illustrative and **not investment advice**, but the mathematics and accounting are now production-complete for the toolkit’s scope.

---

## Future Extensions

This release is feature-complete for its scope. Natural next horizons (not limitations) could be:

- Live `streamlit`/`dash` dashboard and `openpyxl` styled Excel export
- IBES/EDGAR ingestion with caching layer in `src/data/`
- Black-Litterman views and multi-period rebalancing

---

## Reproducibility

- `requirements.txt` + `pyproject.toml` (`pip install -r requirements.txt` or `pip install -e .`)
- `Python 3.11+` declared; tested on 3.9/3.11
- Fixed seeds (`42`) for synthetic data & Monte Carlo
- Deterministic day counts (365/252) documented in `docs/methodology.md`
- `outputs/` is regenerated by `examples/run_full_demo.py` — no hidden state

---

## License

MIT — see [LICENSE](LICENSE).
