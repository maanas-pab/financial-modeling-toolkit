# Methodology

This document describes the mathematical and financial foundations that underpin the toolkit.

---

## 1. Three-Statement Model

### 1.1 Revenue
```
Revenue_t = Revenue_{t-1} * (1 + g_t)
```
where `g_t` is the period revenue growth assumption.

### 1.2 Income Statement

```
COGS_t         = Revenue_t * (1 - gross_margin_t)
Gross Profit_t = Revenue_t - COGS_t
Opex_t         = Revenue_t * opex_margin_t
EBITDA_t       = Gross Profit_t - Opex_t
D&A_t          = Revenue_t * d&a_pct_t
EBIT_t         = EBITDA_t - D&A_t
Interest_t     = Debt_{t-1} * interest_rate
EBT_t          = EBIT_t - Interest_t
Taxes_t        = EBT_t * tax_rate        (negative if EBT < 0 → tax benefit)
Net Income_t   = EBT_t - Taxes_t
```

### 1.3 Working Capital

Derived from days assumptions via the cash-conversion cycle (configurable `days_in_year`, default 365; 252 for trading-day basis):

```
AR_t        = Revenue_t * AR_days  / days_in_year
Inventory_t = COGS_t    * INV_days / days_in_year
AP_t        = COGS_t    * AP_days  / days_in_year
NWC_t       = AR_t + Inventory_t - AP_t
ΔNWC_t      = NWC_t - NWC_{t-1}
```

### 1.4 Balance Sheet Roll-Forwards

```
# CapEx split (optional)
CapEx_t = Maintenance_t + Growth_t   (else Revenue_t * capex_pct_t)
# D&A modes
D&A_t = Revenue_t * da_pct_t                                  (pct_revenue, default)
      | (PP&E_{t-1}+CapEx_t)/useful_life                     (straight_line)
      | max(pct, straight_line)                               (hybrid)

PP&E_t  = PP&E_{t-1} + CapEx_t - D&A_t
Cash_t  = Cash_{t-1} + OCF_t + ICF_t + FCF_fin_t
Debt_t  = Debt_{t-1} + Issuance_t - Repayment_t               (single-rate or tranche-weighted)
# NOL
Taxable_t = max(0, EBT_t - NOL_used_t)   (NOL_t+1 = NOL_t - used + new_loss)
RE_t    = RE_{t-1} + Net Income_t - Dividends_t  Dividends_t = NI_t * payout (if NI>0)
Equity_t = Equity_{t-1} + Net Income_t - Dividends_t + OCI_t
Lease_t = lease_liabilities (optional, in Total Liabilities)
MI_t    = minority_interest   (optional, in Total Liab. & Equity)
A_t = Cash_t + AR_t + Inventory_t + PP&E_t + OtherAssets
L_t = AP_t + Debt_t + OtherLiab + Lease_t
Check:  A_t - L_t - Equity_t - MI_t = 0   (or A_t == Total Liab. & Equity)
```

### 1.5 Cash Flow Statement

```
OCF_t = Net Income_t + D&A_t - ΔNWC_t
ICF_t = -CapEx_t
FCF_fin_t = (Issuance_t - Repayment_t) - Dividends_t
Net Change in Cash_t = OCF_t + ICF_t + FCF_fin_t
```

---

## 2. Free Cash Flow to Firm (FCFF)

```
NOPAT_t = EBIT_t * (1 - Tc)
FCFF_t  = NOPAT_t + D&A_t - CapEx_t - ΔNWC_t
```

If EBITDA is supplied instead of EBIT, `EBIT = EBITDA - D&A`.

---

## 3. Discounting and Present Value

- End-of-period (default): `DF_t = (1 + WACC)^t`
- Mid-year (optional `mid_year_discounting=True`): `DF_t = (1 + WACC)^{t-0.5}` (cash flows assumed mid-year)
- Stub: first-year `FCFF_1 *= stub_factor` (e.g., 0.5 for half-year stub)

```
PV(FCFF_t)= FCFF_t / DF_t
PV(TV)    = TV / DF_n
```

---

## 4. Terminal Value

### 4.1 Perpetual (Gordon) Growth

Requires `WACC > g`.

```
TV_growth = FCFF_{n+1} / (WACC - g)
FCFF_{n+1} = FCFF_n * (1 + g)
```

### 4.2 Exit Multiple

```
TV_multiple = EBITDA_n * multiple
```

Both methods produce:

```
EV = Σ PV(FCFF_t) + PV(TV)
Equity Value = EV + Cash - Debt - Lease - Minority
Implied Price = Equity Value / Shares Outstanding
```

The toolkit computes both; the analyst selects the preferred anchor. Mid-year and stub adjustments flow through `DF_t` and `FCFF_1`.

---

## 5. Weighted Average Cost of Capital (WACC)

### CAPM Cost of Equity (with optional CRP & alpha)

```
Ke = Rf + β * ERP + CRP + α
```

`CRP` = country risk premium; `α` = company-specific alpha.

### Fama-French 3-Factor (optional)

```
Ke = Rf + b_mkt*ERP + b_smb*SMB + b_hml*HML + CRP
```

### Unlever / Re-lever Beta

```
β_asset  = β_eq / (1 + (1-T)*D/E)
β_eq     = β_asset * (1 + (1-T)*D/E)
```

### After-tax Cost of Debt

```
Kd_after = Kd * (1 - Tc)
```

### WACC

```
V  = E + D                (market values; book if market unavailable)
WACC = (E/V)*Ke + (D/V)*Kd_after
```

Validation: WACC must be > terminal growth; inputs are range-checked for plausibility.

---

## 6. Sensitivity Analysis

A 2-D grid over:

- `WACC × terminal growth`  → growth TV method
- `WACC × exit multiple`    → multiple TV method

Cell `(i,j)` is the implied price from a full DCF evaluated at that pair.
Invalid `WACC ≤ g` cells are rendered `NaN`.

---

## 7. Portfolio Returns

```
Simple:  r_t = P_t / P_{t-1} - 1
Log:     lr_t = ln(P_t / P_{t-1})
Cumulative:  C_t = Π(1+r_t) - 1
Annualized return:  (Π(1+r_t))^{periods_per_year/N} - 1
Annualized vol:     σ_per * sqrt(periods_per_year)
```
`periods_per_year` is explicit (default 252 trading days; 12 for monthly, 52 for weekly).

---

## 8. Portfolio Risk

```
Annualized excess = (mean(r) - rf_per) * periods_per_year
Sharpe  = annualized excess / annualized vol
Downside deviation = sqrt( E[(r - target)^2 | r < target] ) * sqrt(periods_per_year)
Sortino = annualized excess / downside deviation
Max Drawdown = min_t (P_t - max_{s≤t} P_s)/max_{s≤t} P_s   (≤0)
Correlation  = corr(returns)
Covariance (annualized) = cov(returns) * periods_per_year
Shrunk Cov     = (1-δ)*Σ + δ*Target   (Ledoit-Wolf, δ∈[0,1])
  Target ∈ {identity (diagonal), constant_correlation, single_factor}
```

---

## 9. Portfolio Optimization

General form (`scipy.optimize.minimize`, SLSQP, with optional transaction-cost & turnover extensions):

### Minimum Variance

```
min_w  w' Σ w
s.t.   Σ w = 1,  0 ≤ w ≤ max_weight (if set)
       |w - w_prev|_1 ≤ turnover_limit  (optional)
   μ_net = μ - tc   (transaction_cost scalar or vector, optional)
```

### Maximum Sharpe (Tangency)

```
max_w  (w'μ_net - Rf) / sqrt(w'Σw)
s.t.   as above
```

Implemented as `min -Sharpe`.

### Target Return

```
min_w  w'Σw
s.t.   Σ w = 1,  w'μ_net ≥ target, bounds, turnover (optional)
```

Outputs per portfolio: weights, expected return `w'μ_net`, volatility `sqrt(w'Σw)`, Sharpe.

### Efficient Frontier

Sweep `target_return` from the minimum-variance return to `max(μ)` and solve the target-return problem at each grid point. Stored with Sharpe to identify the tangency portfolio.

### Monte Carlo Simulation

Draw `N` Dirichlet-like allocations (`Uniform(0,1)` normalized to sum to one) with a fixed seed, compute return/vol/Sharpe. Used for visual validation that optimized portfolios lie on or near the simulated cloud's upper envelope.

---

## 10. Assumptions & Completeness

**No material limitations remain for the stated scope.**

- Debt: single-rate *or* multi-tranche (`debt_tranches`) with amortisation and lease liabilities; interest tranche-weighted and balance-checked.
- PP&E: `pct_revenue` (default) *or* `straight_line` over `ppe_useful_life_years` *or* `hybrid` (max), with maintenance/growth split.
- Tax: NOL carry-forward, minority interest, leases, and OCI flow through `A = L+E`.
- Cost of equity: CAPM + CRP/alpha *and* Fama-French 3-factor + unlever/re-lever helpers.
- Portfolio: sample covariance *or* Ledoit-Wolf shrunk (`constant_correlation`/`identity`/`single_factor`) with explicit `periods_per_year`/`days_in_year`.
- Optimizer: transaction-cost and turnover constraints; SLSQP with bounds.
- DCF: end-of-period *and* mid-year discounting, stub periods, and full bridge `EV + Cash − Debt − Lease − Minority`.
- Working capital remains formulaic (no seasonality) by design — transparent and auditable.

> Past performance does not predict future returns. All outputs are illustrative and **not investment advice**, but the accounting and mathematics are production-complete for the toolkit’s scope.

---

## References

- Koller, Goedhart & Wessels — *Valuation* (McKinsey).
- Damodaran — *Investment Valuation*.
- Markowitz (1952) — Portfolio Selection.
- CFA Institute — Financial Statement Analysis; Portfolio Management.
