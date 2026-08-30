"""DCF valuation engine — perpetual growth and exit-multiple methods."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DCFInputs:
    """
    Inputs for the DCF model.

    Each list is ordered by forecast year (year 1 = next year).
    Scalars are broadcast; but lists are preferred for clarity.

    Attributes:
        revenue: Forecast revenue per year.
        ebitda: Forecast EBITDA per year (optional if ebit given).
        ebit: Forecast EBIT per year.
        da: D&A per year.
        capex: CapEx per year.
        working_capital_change: Change in NWC per year (positive = outflow).
        tax_rate: Marginal tax rate.
        wacc: Discount rate.
        terminal_growth: Perpetual growth for terminal value.
        exit_multiple: EV/EBITDA exit multiple (for alternative method).
        debt: Current total debt (for equity value).
        cash: Current cash & equivalents.
        shares_outstanding: Fully diluted shares.
        forecast_years: Inferred from revenue if not specified.
    """

    revenue: list[float]
    ebit: list[float] | None = None
    ebitda: list[float] | None = None
    da: list[float] | None = None
    capex: list[float] | None = None
    working_capital_change: list[float] | None = None
    tax_rate: float = 0.25
    wacc: float = 0.09
    terminal_growth: float = 0.02
    exit_multiple: float = 10.0
    debt: float = 0.0
    cash: float = 0.0
    shares_outstanding: float = 10.0
    # Extended: enterprise-value bridge completeness
    lease_liabilities: float = 0.0
    minority_interest: float = 0.0
    mid_year_discounting: bool = False
    stub_period_factor: float = 1.0  # 0-1 for stub first year (1 = full year)

    def __post_init__(self) -> None:
        n = len(self.revenue)
        if n == 0:
            raise ValueError("revenue must be non-empty")
        if self.ebit is None and self.ebitda is None:
            raise ValueError("Provide at least one of ebit or ebitda")
        if self.ebit is not None and len(self.ebit) != n:
            raise ValueError("ebit length must match revenue")
        if self.ebitda is not None and len(self.ebitda) != n:
            raise ValueError("ebitda length must match revenue")
        if self.da is not None and len(self.da) != n:
            raise ValueError("da length must match revenue")
        if self.capex is not None and len(self.capex) != n:
            raise ValueError("capex length must match revenue")
        if self.working_capital_change is not None and len(self.working_capital_change) != n:
            raise ValueError("working_capital_change length must match revenue")
        if not 0 <= self.tax_rate < 1:
            raise ValueError("tax_rate must be in [0, 1)")
        if self.wacc <= 0:
            raise ValueError("wacc must be > 0")
        if self.wacc <= self.terminal_growth:
            raise ValueError(f"wacc ({self.wacc}) must be > terminal_growth ({self.terminal_growth}) for Gordon growth")
        if self.shares_outstanding <= 0:
            raise ValueError("shares_outstanding must be > 0")
        if self.debt < 0 or self.cash < 0:
            raise ValueError("debt and cash must be >= 0")
        if self.lease_liabilities < 0 or self.minority_interest < 0:
            raise ValueError("lease_liabilities and minority_interest must be >= 0")
        if not 0 < self.stub_period_factor <= 1:
            raise ValueError("stub_period_factor must be in (0,1]")


class DCFModel:
    """
    Free Cash Flow to Firm (FCFF) DCF model.

    FCFF = NOPAT + D&A - CapEx - Change in NWC
         = EBIT*(1 - Tc) + D&A - CapEx - dWC

    Supports both:
      - Perpetual growth terminal value: TV = FCFF_{n+1} / (WACC - g)
      - Exit-multiple terminal value: TV = EBITDA_n * Multiple

    All PV computations use end-of-period discounting.
    """

    def __init__(self, inputs: DCFInputs) -> None:
        self.inputs = inputs
        self._results: pd.DataFrame | None = None
        self._summary: dict | None = None

    # ------------------------------------------------------------------ #
    def _fcff_series(self) -> tuple[list[float], list[float]]:
        """Return (FCFF list, NOPAT list)."""
        inp = self.inputs
        n = len(inp.revenue)
        if inp.ebit is not None:
            ebit = inp.ebit
        else:
            assert inp.ebitda is not None and inp.da is not None
            ebit = [eb - d for eb, d in zip(inp.ebitda, inp.da)]

        nopat = [e * (1 - inp.tax_rate) for e in ebit]
        da = inp.da if inp.da is not None else [0.0] * n
        capex = inp.capex if inp.capex is not None else [0.0] * n
        dwc = inp.working_capital_change if inp.working_capital_change is not None else [0.0] * n

        fcff = [nopat[i] + da[i] - capex[i] - dwc[i] for i in range(n)]
        return fcff, nopat, ebit  # type: ignore

    # ------------------------------------------------------------------ #
    def build(self) -> pd.DataFrame:
        """Build the full FCFF schedule and return it."""
        inp = self.inputs
        fcff, nopat, ebit = self._fcff_series()
        # Apply stub period to first-year FCFF if needed
        if inp.stub_period_factor != 1.0:
            fcff[0] = fcff[0] * inp.stub_period_factor
        n = len(fcff)
        # Discount factors (mid-year optional)
        if inp.mid_year_discounting:
            df_factors = [(1 + inp.wacc) ** (i + 0.5) for i in range(n)]
        else:
            df_factors = [(1 + inp.wacc) ** (i + 1) for i in range(n)]
        pv_fcff = [f / d for f, d in zip(fcff, df_factors)]

        # Terminal values — both methods for comparison
        # Perpetual growth
        fcff_next = fcff[-1] * (1 + inp.terminal_growth)
        tv_growth = fcff_next / (inp.wacc - inp.terminal_growth)
        pv_tv_growth = tv_growth / df_factors[-1]

        # Exit multiple
        if inp.ebitda is not None:
            ebitda_last = inp.ebitda[-1]
        elif inp.ebit is not None and inp.da is not None:
            ebitda_last = inp.ebit[-1] + inp.da[-1]
        else:
            ebitda_last = ebit[-1]  # fallback

        tv_multiple = ebitda_last * inp.exit_multiple
        pv_tv_multiple = tv_multiple / df_factors[-1]

        ev_growth = sum(pv_fcff) + pv_tv_growth
        ev_multiple = sum(pv_fcff) + pv_tv_multiple

        eq_growth = ev_growth + inp.cash - inp.debt - inp.lease_liabilities - inp.minority_interest
        eq_multiple = ev_multiple + inp.cash - inp.debt - inp.lease_liabilities - inp.minority_interest

        price_growth = eq_growth / inp.shares_outstanding
        price_multiple = eq_multiple / inp.shares_outstanding

        self._summary = {
            "pv_fcff": sum(pv_fcff),
            "tv_growth": tv_growth,
            "pv_tv_growth": pv_tv_growth,
            "tv_multiple": tv_multiple,
            "pv_tv_multiple": pv_tv_multiple,
            "enterprise_value_growth": ev_growth,
            "enterprise_value_multiple": ev_multiple,
            "equity_value_growth": eq_growth,
            "equity_value_multiple": eq_multiple,
            "implied_price_growth": price_growth,
            "implied_price_multiple": price_multiple,
        }

        df = pd.DataFrame(
            {
                "Revenue": inp.revenue,
                "EBIT": ebit,
                "NOPAT": nopat,
                "D&A": inp.da if inp.da is not None else [0.0] * n,
                "CapEx": inp.capex if inp.capex is not None else [0.0] * n,
                "WC Change": inp.working_capital_change if inp.working_capital_change is not None else [0.0] * n,
                "FCFF": fcff,
                "Discount Factor": df_factors,
                "PV of FCFF": pv_fcff,
            },
            index=[f"Year {i+1}" for i in range(n)],
        ).T
        self._results = df
        return df

    # ------------------------------------------------------------------ #
    def summary(self, method: str = "growth") -> dict:
        """
        Return headline valuation summary.

        Args:
            method: 'growth' or 'multiple'
        """
        if self._summary is None:
            self.build()
        assert self._summary is not None
        s = self._summary
        if method == "growth":
            return {
                "method": "Perpetual Growth",
                "pv_of_fcff": s["pv_fcff"],
                "terminal_value": s["tv_growth"],
                "pv_of_terminal_value": s["pv_tv_growth"],
                "enterprise_value": s["enterprise_value_growth"],
                "equity_value": s["equity_value_growth"],
                "implied_share_price": s["implied_price_growth"],
            }
        elif method == "multiple":
            return {
                "method": "Exit Multiple",
                "pv_of_fcff": s["pv_fcff"],
                "terminal_value": s["tv_multiple"],
                "pv_of_terminal_value": s["pv_tv_multiple"],
                "enterprise_value": s["enterprise_value_multiple"],
                "equity_value": s["equity_value_multiple"],
                "implied_share_price": s["implied_price_multiple"],
            }
        else:
            raise ValueError("method must be 'growth' or 'multiple'")

    def implied_price(self, method: str = "growth") -> float:
        return self.summary(method)["implied_share_price"]

    def to_dataframe(self) -> pd.DataFrame:
        if self._results is None:
            self.build()
        assert self._results is not None
        return self._results

    @classmethod
    def from_three_statement(
        cls,
        model,
        wacc: float,
        terminal_growth: float = 0.02,
        exit_multiple: float = 10.0,
        mid_year: bool = False,
        stub_factor: float = 1.0,
    ) -> "DCFModel":
        """
        Build a DCFModel directly from a ThreeStatementModel instance.
        Includes lease/minority bridge if present in assumptions.
        """
        assert model.income_statement is not None
        assert model.cash_flow is not None
        assert model.balance_sheet is not None
        rev = model.income_statement.loc["Revenue"].tolist()
        ebit = model.income_statement.loc["EBIT"].tolist()
        da = model.income_statement.loc["D&A"].tolist()
        # CapEx: negative of CFS CapEx line (which is stored as negative)
        capex_signed = model.cash_flow.loc["CapEx"].tolist()  # negative values
        capex = [-c for c in capex_signed]  # positive cost
        wc_change = model.cash_flow.loc["Working Capital Change"].tolist()
        a = model.assumptions
        inputs = DCFInputs(
            revenue=rev,
            ebit=ebit,
            da=da,
            capex=capex,
            working_capital_change=wc_change,
            tax_rate=a.tax_rate,
            wacc=wacc,
            terminal_growth=terminal_growth,
            exit_multiple=exit_multiple,
            debt=a.beginning_debt,
            cash=a.beginning_cash,
            shares_outstanding=a.share_count,
            lease_liabilities=getattr(a, "lease_liabilities", 0.0),
            minority_interest=getattr(a, "minority_interest", 0.0),
            mid_year_discounting=mid_year,
            stub_period_factor=stub_factor,
        )
        return cls(inputs)
