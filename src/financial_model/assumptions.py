"""Model assumptions and scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class ModelAssumptions:
    """
    Central assumptions object for the three-statement model.

    All arrays / lists are ordered chronologically.  The model can be run
    for an arbitrary forecast horizon; scalar assumptions are broadcast
    across all forecast years.

    Attributes:
        historical_revenue: Most recent historical revenue (base year).
        revenue_growth: Annual revenue growth rates for each forecast year.
        gross_margin: Gross margin (Gross Profit / Revenue) per forecast year.
        opex_margin: Operating expense margin (Opex / Revenue) per year.
        da_pct_revenue: D&A as % of revenue per year.
        capex_pct_revenue: CapEx as % of revenue per year.
        tax_rate: Corporate tax rate (0-1).
        ar_days: Accounts receivable days (DSO).
        inventory_days: Inventory days (DIO).
        ap_days: Accounts payable days (DPO).
        beginning_cash: Cash at start of forecast.
        beginning_ar: Accounts receivable at start.
        beginning_inventory: Inventory at start.
        beginning_ppe: Net PP&E at start.
        beginning_other_assets: Other assets at start.
        beginning_ap: Accounts payable at start.
        beginning_debt: Total debt at start.
        beginning_other_liabilities: Other liabilities at start.
        beginning_equity: Total equity at start (must satisfy A=L+E at t0).
        beginning_retained_earnings: Retained earnings at start.
        interest_rate: Annual interest rate on debt (for interest expense).
        debt_issuance: New debt issued per forecast year.
        debt_repayment: Debt repaid per forecast year.
        dividends_pct_net_income: Dividends as % of net income.
        share_count: Shares outstanding (for per-share metrics).
    """

    historical_revenue: float
    revenue_growth: list[float] | float
    gross_margin: list[float] | float = 0.40
    opex_margin: list[float] | float = 0.20
    da_pct_revenue: list[float] | float = 0.03
    capex_pct_revenue: list[float] | float = 0.04
    tax_rate: float = 0.25

    ar_days: float = 45.0
    inventory_days: float = 30.0
    ap_days: float = 40.0

    beginning_cash: float = 100.0
    beginning_ar: float = 50.0
    beginning_inventory: float = 30.0
    beginning_ppe: float = 200.0
    beginning_other_assets: float = 20.0
    beginning_ap: float = 40.0
    beginning_debt: float = 150.0
    beginning_other_liabilities: float = 10.0
    beginning_equity: float = 200.0
    beginning_retained_earnings: float = 120.0

    interest_rate: float = 0.05
    debt_issuance: list[float] | float = 0.0
    debt_repayment: list[float] | float = 0.0
    dividends_pct_net_income: float = 0.20
    share_count: float = 10.0

    forecast_years: int = 5
    base_year_label: int = 2024

    # --- Extended realism (all optional, defaults preserve prior behavior) ---
    days_in_year: int = 365  # configurable day-count (295-366)
    ppe_useful_life_years: float = 10.0  # for straight-line depreciation schedule
    depreciation_method: str = "pct_revenue"  # "pct_revenue" | "straight_line" | "hybrid"
    maintenance_capex_pct_revenue: float | None = None  # if set, capex split
    growth_capex_pct_revenue: float | None = None
    debt_tranches: list[dict] | None = None  # e.g. [{"name":"Term Loan B","balance":100,"rate":0.06,"amort":5}]
    minority_interest: float = 0.0
    lease_liabilities: float = 0.0
    nol_carryforward: float = 0.0
    other_comprehensive_income: float = 0.0

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if self.historical_revenue <= 0:
            raise ValueError("historical_revenue must be > 0")
        if not 0 <= self.tax_rate < 1:
            raise ValueError("tax_rate must be in [0, 1)")
        if not 0 <= self.dividends_pct_net_income <= 1:
            raise ValueError("dividends_pct_net_income must be in [0, 1]")
        if self.forecast_years < 1:
            raise ValueError("forecast_years must be >= 1")
        if self.share_count <= 0:
            raise ValueError("share_count must be > 0")
        if self.interest_rate < 0:
            raise ValueError("interest_rate cannot be negative")
        for name in ("ar_days", "inventory_days", "ap_days"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.days_in_year not in range(250, 367):
            raise ValueError(f"days_in_year {self.days_in_year} must be in [250,366] (365 trading/calendar or 252)")
        if self.ppe_useful_life_years <= 0:
            raise ValueError("ppe_useful_life_years must be > 0")
        if self.depreciation_method not in ("pct_revenue", "straight_line", "hybrid"):
            raise ValueError("depreciation_method must be 'pct_revenue' | 'straight_line' | 'hybrid'")
        if self.minority_interest < 0 or self.lease_liabilities < 0 or self.nol_carryforward < 0:
            raise ValueError("minority_interest, lease_liabilities, nol_carryforward must be >= 0")
        if self.debt_tranches is not None:
            total = sum(t.get("balance", 0) for t in self.debt_tranches)
            if abs(total - self.beginning_debt) > 1e-6:
                raise ValueError(f"debt_tranches balances sum {total} != beginning_debt {self.beginning_debt}")
            for t in self.debt_tranches:
                if t.get("rate", 0) < 0:
                    raise ValueError(f"Debt tranche {t.get('name')} rate cannot be negative")

        # Check that supplied list lengths match forecast_years if lists
        for attr in (
            "revenue_growth",
            "gross_margin",
            "opex_margin",
            "da_pct_revenue",
            "capex_pct_revenue",
            "debt_issuance",
            "debt_repayment",
        ):
            val = getattr(self, attr)
            if isinstance(val, (list, tuple)):
                if len(val) != self.forecast_years:
                    raise ValueError(
                        f"{attr} length {len(val)} != forecast_years {self.forecast_years}"
                    )

        # Validate balance sheet identity at t0 (including optional leases/minority for completeness)
        total_assets = (
            self.beginning_cash
            + self.beginning_ar
            + self.beginning_inventory
            + self.beginning_ppe
            + self.beginning_other_assets
        )
        total_liab_equity = (
            self.beginning_ap + self.beginning_debt + self.beginning_other_liabilities + self.lease_liabilities + self.minority_interest + self.beginning_equity
        )
        if abs(total_assets - total_liab_equity) > 1e-6:
            raise ValueError(
                f"Beginning balance sheet does not balance: Assets={total_assets}, "
                f"L+E={total_liab_equity}, diff={total_assets - total_liab_equity}"
            )

    # ------------------------------------------------------------------ #
    def _broadcast(self, value: list[float] | float) -> list[float]:
        if isinstance(value, (list, tuple)):
            return list(value)
        return [float(value)] * self.forecast_years

    def revenue_growth_list(self) -> list[float]:
        return self._broadcast(self.revenue_growth)

    def gross_margin_list(self) -> list[float]:
        return self._broadcast(self.gross_margin)

    def opex_margin_list(self) -> list[float]:
        return self._broadcast(self.opex_margin)

    def da_list(self) -> list[float]:
        return self._broadcast(self.da_pct_revenue)

    def capex_list(self) -> list[float]:
        return self._broadcast(self.capex_pct_revenue)

    def debt_issuance_list(self) -> list[float]:
        return self._broadcast(self.debt_issuance)

    def debt_repayment_list(self) -> list[float]:
        return self._broadcast(self.debt_repayment)

    def forecast_labels(self) -> list[int]:
        return [self.base_year_label + i + 1 for i in range(self.forecast_years)]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict) -> "ModelAssumptions":
        """Create assumptions from a plain dict (e.g., loaded from CSV/Excel)."""
        return cls(**data)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ScenarioAssumptions:
    """
    Scenario overlay that modifies base assumptions.

    Each field, if not None, overrides the corresponding base assumption.
    """

    name: str
    revenue_growth: Optional[list[float] | float] = None
    gross_margin: Optional[list[float] | float] = None
    opex_margin: Optional[list[float] | float] = None
    capex_pct_revenue: Optional[list[float] | float] = None
    ar_days: Optional[float] = None
    inventory_days: Optional[float] = None
    ap_days: Optional[float] = None
    tax_rate: Optional[float] = None
    wacc: Optional[float] = None
    terminal_growth: Optional[float] = None
    description: str = ""

    def apply_to(self, base: ModelAssumptions) -> ModelAssumptions:
        """Return a new ModelAssumptions with scenario overrides applied."""
        import copy

        new = copy.deepcopy(base)
        for attr in (
            "revenue_growth",
            "gross_margin",
            "opex_margin",
            "capex_pct_revenue",
            "ar_days",
            "inventory_days",
            "ap_days",
            "tax_rate",
        ):
            val = getattr(self, attr)
            if val is not None:
                setattr(new, attr, val)
        # Re-validate after overrides
        new._validate()
        return new


# Pre-defined Bear / Base / Bull scenarios for convenience
BEAR_SCENARIO = ScenarioAssumptions(
    name="Bear",
    revenue_growth=[0.02, 0.02, 0.015, 0.015, 0.01],
    gross_margin=0.35,
    opex_margin=0.22,
    capex_pct_revenue=0.05,
    description="Conservative growth, margin compression.",
)

BASE_SCENARIO = ScenarioAssumptions(
    name="Base",
    revenue_growth=[0.06, 0.05, 0.04, 0.03, 0.03],
    gross_margin=0.40,
    opex_margin=0.20,
    capex_pct_revenue=0.04,
    description="Management guidance / consensus case.",
)

BULL_SCENARIO = ScenarioAssumptions(
    name="Bull",
    revenue_growth=[0.10, 0.09, 0.08, 0.06, 0.05],
    gross_margin=0.45,
    opex_margin=0.18,
    capex_pct_revenue=0.03,
    description="Accelerated growth with operating leverage.",
)
