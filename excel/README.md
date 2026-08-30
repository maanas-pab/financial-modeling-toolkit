# Excel Companion

The `excel/` folder is intended to hold the companion Excel workbook that mirrors the Python toolkit logic. Keeping the two implementations side-by-side allows:

- Cross-validation: the same assumptions run in both Excel and Python should produce identical statements and valuation.
- Accessibility: Excel remains the lingua franca of investment banking and corporate finance; Python provides reproducibility.

## Recommended Workbook Structure

| Sheet | Purpose |
|---|---|
| `Assumptions` | Revenue growth, margins, D&A%, CapEx%, NWC days, tax, debt, rates, shares |
| `Income Statement` | Forecast IS linked to Assumptions |
| `Balance Sheet` | Forecast BS with cash as plug |
| `Cash Flow` | OCF / ICF / FCF and cash reconciliation |
| `Checks` | A=L+E, cash, debt, RE checks (pass/fail) |
| `DCF` | FCFF, TV (growth + multiple), PV, EV → Price |
| `WACC` | Ke (CAPM), Kd*(1-T), weights, WACC |
| `Sensitivity` | WACC×g and WACC×Multiple data tables |

## Python ↔ Excel Integration

The Python engine can **read** assumptions from Excel and **write** results back:

```python
from src.data.loader import load_assumptions_from_excel, export_results_to_excel
from src.financial_model.three_statement import ThreeStatementModel

assumptions = load_assumptions_from_excel("excel/model.xlsx", sheet_name="Assumptions")
model = ThreeStatementModel(assumptions)

export_results_to_excel(
    "outputs/results.xlsx",
    income_statement=model.income_statement,
    balance_sheet=model.balance_sheet,
    cash_flow=model.cash_flow,
)
```

The Python financial engine remains independent — it never *requires* Excel to run. The Excel workbook and Python code are two implementations of the same analytical framework.

## Template

If no workbook exists yet, create one by exporting the Python model:

```python
model.to_excel("excel/model_template.xlsx")
```

Then edit `assumptions` in Excel and reload via `load_assumptions_from_excel`.
