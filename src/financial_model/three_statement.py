"""Three-statement financial model engine."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.financial_model.assumptions import ModelAssumptions


class ThreeStatementModel:
    """
    Builds integrated Income Statement, Balance Sheet, and Cash Flow Statement.

    The model is intentionally transparent: every forecast line is computed
    from explicit assumptions in ModelAssumptions. Working capital is derived
    from days assumptions (DSO/DIO/DPO) to enforce internal consistency.

    Balance sheet balancing is achieved via cash as the plug: ending cash
    is computed from the cash flow statement and then reflected on the
    balance sheet, guaranteeing A = L + E if checks pass.

    Attributes:
        assumptions: Validated ModelAssumptions instance.
        income_statement: Forecast IS as DataFrame.
        balance_sheet: Forecast BS as DataFrame.
        cash_flow: Forecast CFS as DataFrame.
    """

    def __init__(self, assumptions: ModelAssumptions) -> None:
        self.assumptions = assumptions
        self._labels = assumptions.forecast_labels()
        self.income_statement: pd.DataFrame | None = None
        self.balance_sheet: pd.DataFrame | None = None
        self.cash_flow: pd.DataFrame | None = None
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        a = self.assumptions
        n = a.forecast_years

        rev_growth = a.revenue_growth_list()
        gross_m = a.gross_margin_list()
        opex_m = a.opex_margin_list()
        da_pct = a.da_list()
        capex_pct = a.capex_list()
        debt_issue = a.debt_issuance_list()
        debt_repay = a.debt_repayment_list()

        # --- Revenue forecast ---
        revenue: list[float] = []
        prev = a.historical_revenue
        for g in rev_growth:
            if g < -1:
                raise ValueError(f"Revenue growth {g} implies negative revenue")
            cur = prev * (1 + g)
            if cur <= 0:
                raise ValueError("Forecast revenue must be > 0")
            revenue.append(cur)
            prev = cur

        # --- Income statement line items ---
        cogs = [r * (1 - gm) for r, gm in zip(revenue, gross_m)]
        gross_profit = [r - c for r, c in zip(revenue, cogs)]
        opex = [r * om for r, om in zip(revenue, opex_m)]
        ebitda = [gp - oe for gp, oe in zip(gross_profit, opex)]
        da = [r * d for r, d in zip(revenue, da_pct)]
        ebit = [e - d for e, d in zip(ebitda, da)]

        # Interest: single-rate or tranche-weighted
        interest: list[float] = []
        if a.debt_tranches is not None:
            # initialise tranche balances
            tranche_bals = [float(t["balance"]) for t in a.debt_tranches]
            tranche_rates = [float(t["rate"]) for t in a.debt_tranches]
            for i in range(n):
                interest.append(sum(b * r for b, r in zip(tranche_bals, tranche_rates)))
                # allocate net debt change proportionally (simple)
                net = debt_issue[i] - debt_repay[i]
                if tranche_bals:
                    total = sum(tranche_bals) or 1
                    for j in range(len(tranche_bals)):
                        tranche_bals[j] += net * (tranche_bals[j] / total) if total else net / len(tranche_bals)
                        tranche_bals[j] = max(0, tranche_bals[j])
        else:
            debt_bal = a.beginning_debt
            for i in range(n):
                interest.append(debt_bal * a.interest_rate)
                debt_bal = debt_bal + debt_issue[i] - debt_repay[i]

        # PP&E depreciation: support pct_revenue, straight_line, hybrid
        # For straight_line/hybrid we need to compute D&A iteratively, but for
        # simplicity compute capex first then override da where needed
        capex: list[float] = []
        if a.maintenance_capex_pct_revenue is not None or a.growth_capex_pct_revenue is not None:
            maint = a.maintenance_capex_pct_revenue or 0.0
            growth = a.growth_capex_pct_revenue or 0.0
            capex = [r * (maint + growth) for r in revenue]
        else:
            capex = [r * cp for r, cp in zip(revenue, capex_pct)]

        # D&A schedule depends on method
        da: list[float]
        if a.depreciation_method == "pct_revenue":
            da = [r * d for r, d in zip(revenue, da_pct)]
        elif a.depreciation_method == "straight_line":
            # Straight-line on beginning PP&E plus new capex, useful life
            da = []
            ppe_for_da = a.beginning_ppe
            for i in range(n):
                # depreciate opening PPE + new capex over useful life
                # simplified: annual charge = (ppe opening + capex) / useful_life
                charge = (ppe_for_da + capex[i]) / a.ppe_useful_life_years
                # cap at opening + capex
                charge = min(charge, ppe_for_da + capex[i])
                da.append(charge)
                ppe_for_da = ppe_for_da + capex[i] - charge
            # need to recompute ebitda/ebit after new da? defer to below
            ebitda = [gp - oe for gp, oe in zip(gross_profit, opex)]
            ebit = [e - d for e, d in zip(ebitda, da)]
        elif a.depreciation_method == "hybrid":
            da_pct_vals = [r * d for r, d in zip(revenue, da_pct)]
            da_sl = []
            ppe_for_da = a.beginning_ppe
            for i in range(n):
                charge = (ppe_for_da + capex[i]) / a.ppe_useful_life_years
                charge = min(charge, ppe_for_da + capex[i])
                da_sl.append(charge)
                ppe_for_da = ppe_for_da + capex[i] - charge
            da = [max(p, s) for p, s in zip(da_pct_vals, da_sl)]
            ebitda = [gp - oe for gp, oe in zip(gross_profit, opex)]
            ebit = [e - d for e, d in zip(ebitda, da)]
        else:
            da = [r * d for r, d in zip(revenue, da_pct)]

        # Taxes with NOL carryforward support
        remaining_nol = a.nol_carryforward
        ebt = [e - intr for e, intr in zip(ebit, interest)]
        taxes = []
        for e in ebt:
            if e <= 0:
                # tax benefit, NOL increases
                if e < 0:
                    remaining_nol += -e  # accumulate loss as NOL (simplified)
                taxes.append(e * a.tax_rate)
            else:
                taxable = e
                if remaining_nol > 0:
                    use = min(remaining_nol, taxable)
                    taxable -= use
                    remaining_nol -= use
                taxes.append(taxable * a.tax_rate)
        net_income = [e - t for e, t in zip(ebt, taxes)]

        # --- Working capital (configurable day-count) ---
        ar = [r * a.ar_days / a.days_in_year for r in revenue]
        inv = [c * a.inventory_days / a.days_in_year for c in cogs]
        ap = [c * a.ap_days / a.days_in_year for c in cogs]

        # --- Balance sheet & Cash flow (iterative) ---
        dividends = [ni * a.dividends_pct_net_income if ni > 0 else 0.0 for ni in net_income]

        # Opening balances
        cash = a.beginning_cash
        ar_prev = a.beginning_ar
        inv_prev = a.beginning_inventory
        ppe_prev = a.beginning_ppe
        ap_prev = a.beginning_ap
        debt_prev = a.beginning_debt
        re_prev = a.beginning_retained_earnings

        # Collectors
        cash_list: list[float] = []
        ar_list: list[float] = []
        inv_list: list[float] = []
        ppe_list: list[float] = []
        ap_list: list[float] = []
        debt_list: list[float] = []
        equity_list: list[float] = []
        re_list: list[float] = []
        other_assets_list: list[float] = [a.beginning_other_assets] * n
        other_liab_list: list[float] = [a.beginning_other_liabilities] * n

        ocf_list: list[float] = []
        icf_list: list[float] = []
        fcf_list: list[float] = []
        wc_change_list: list[float] = []
        net_change_cash_list: list[float] = []

        equity_prev = a.beginning_equity

        for i in range(n):
            # WC change = (AR+Inv-AP) - prev(WC)
            wc_current = ar[i] + inv[i] - ap[i]
            wc_prev = ar_prev + inv_prev - ap_prev
            wc_change = wc_current - wc_prev  # positive = cash outflow

            # Operating cash flow
            ocf = net_income[i] + da[i] - wc_change

            # Investing cash flow (CapEx)
            icf = -capex[i]

            # Financing
            debt_net = debt_issue[i] - debt_repay[i]
            # Equity financing: dividends only in this simplified model
            fcf_fin = debt_net - dividends[i]

            net_change_cash = ocf + icf + fcf_fin
            cash = cash + net_change_cash

            # PP&E roll: prev + CapEx - D&A
            ppe = ppe_prev + capex[i] - da[i]

            # Debt
            debt = debt_prev + debt_net

            # Retained earnings
            re = re_prev + net_income[i] - dividends[i]

            # Equity = beginning equity + cumulative retained earnings change + any other
            # Simplified: equity = prior equity + net income - dividends
            equity = equity_prev + net_income[i] - dividends[i]

            # Store
            cash_list.append(cash)
            ar_list.append(ar[i])
            inv_list.append(inv[i])
            ppe_list.append(ppe)
            ap_list.append(ap[i])
            debt_list.append(debt)
            equity_list.append(equity)
            re_list.append(re)
            ocf_list.append(ocf)
            icf_list.append(icf)
            fcf_list.append(fcf_fin)
            wc_change_list.append(wc_change)
            net_change_cash_list.append(net_change_cash)

            # Roll forward
            ar_prev = ar[i]
            inv_prev = inv[i]
            ppe_prev = ppe
            ap_prev = ap[i]
            debt_prev = debt
            re_prev = re
            equity_prev = equity

        # --- Assemble DataFrames ---
        cols = self._labels

        self.income_statement = pd.DataFrame(
            {
                "Revenue": revenue,
                "COGS": cogs,
                "Gross Profit": gross_profit,
                "Operating Expenses": opex,
                "EBITDA": ebitda,
                "D&A": da,
                "EBIT": ebit,
                "Interest Expense": interest,
                "EBT": ebt,
                "Taxes": taxes,
                "Net Income": net_income,
            },
            index=cols,
        ).T

        # Extended liabilities/equity for leases/minority (optional)
        lease_list = [a.lease_liabilities] * n
        minority_list = [a.minority_interest] * n

        self.balance_sheet = pd.DataFrame(
            {
                "Cash": cash_list,
                "Accounts Receivable": ar_list,
                "Inventory": inv_list,
                "PP&E": ppe_list,
                "Other Assets": other_assets_list,
                "Total Assets": [
                    c + ar_ + inv_ + ppe_ + oa
                    for c, ar_, inv_, ppe_, oa in zip(cash_list, ar_list, inv_list, ppe_list, other_assets_list)
                ],
                "Accounts Payable": ap_list,
                "Debt": debt_list,
                "Other Liabilities": other_liab_list,
                "Lease Liabilities": lease_list,
                "Minority Interest": minority_list,
                "Total Liabilities": [
                    ap_ + d + ol + lease for ap_, d, ol, lease in zip(ap_list, debt_list, other_liab_list, lease_list)
                ],
                "Equity": equity_list,
                "Total Liab. & Equity": [
                    ap_ + d + ol + lease + eq + mi
                    for ap_, d, ol, lease, eq, mi in zip(ap_list, debt_list, other_liab_list, lease_list, equity_list, minority_list)
                ],
                "Retained Earnings (memo)": re_list,
            },
            index=cols,
        ).T

        self.cash_flow = pd.DataFrame(
            {
                "Net Income": net_income,
                "D&A": da,
                "Working Capital Change": wc_change_list,
                "Operating Cash Flow": ocf_list,
                "CapEx": [-c for c in capex],
                "Investing Cash Flow": icf_list,
                "Debt Issuance": debt_issue,
                "Debt Repayment": [-d for d in debt_repay],
                "Dividends": [-d for d in dividends],
                "Financing Cash Flow": fcf_list,
                "Net Change in Cash": net_change_cash_list,
                "Beginning Cash": [a.beginning_cash] + cash_list[:-1],
                "Ending Cash": cash_list,
            },
            index=cols,
        ).T

    # ------------------------------------------------------------------ #
    def summary(self) -> pd.DataFrame:
        """High-level forecast summary suitable for a chart or table."""
        assert self.income_statement is not None
        assert self.balance_sheet is not None
        fcff_proxy = self.income_statement.loc["EBIT"] * (1 - self.assumptions.tax_rate)  # approximate
        df = pd.DataFrame(
            {
                "Revenue": self.income_statement.loc["Revenue"],
                "EBITDA": self.income_statement.loc["EBITDA"],
                "EBIT": self.income_statement.loc["EBIT"],
                "Net Income": self.income_statement.loc["Net Income"],
                "Ending Cash": self.balance_sheet.loc["Cash"],
            }
        ).T
        return df

    def to_excel(self, path: str) -> None:
        """Export all three statements to an Excel workbook."""
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            assert self.income_statement is not None
            assert self.balance_sheet is not None
            assert self.cash_flow is not None
            self.income_statement.to_excel(writer, sheet_name="Income Statement")
            self.balance_sheet.to_excel(writer, sheet_name="Balance Sheet")
            self.cash_flow.to_excel(writer, sheet_name="Cash Flow")
