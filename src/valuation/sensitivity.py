"""Sensitivity analysis and scenario engine."""

from __future__ import annotations

import itertools
import pandas as pd
import numpy as np

from src.valuation.dcf import DCFInputs, DCFModel
from src.financial_model.assumptions import ModelAssumptions, ScenarioAssumptions


def _clone_inputs_with_overrides(
    base: DCFInputs, wacc: float | None = None, terminal_growth: float | None = None, exit_multiple: float | None = None
) -> DCFInputs:
    import copy

    new = copy.deepcopy(base)
    if wacc is not None:
        new.wacc = wacc
    if terminal_growth is not None:
        new.terminal_growth = terminal_growth
    if exit_multiple is not None:
        new.exit_multiple = exit_multiple
    # Re-validate WACC vs terminal growth
    if new.wacc <= new.terminal_growth:
        raise ValueError(f"wacc {new.wacc} must be > terminal_growth {new.terminal_growth}")
    return new


def sensitivity_wacc_growth(
    base_inputs: DCFInputs,
    wacc_range: list[float],
    growth_range: list[float],
    method: str = "growth",
) -> pd.DataFrame:
    """
    2-D sensitivity: WACC (rows) x Terminal Growth (columns).

    Returns DataFrame of implied share price.
    """
    data = []
    for w in wacc_range:
        row: list[float] = []
        for g in growth_range:
            try:
                inp = _clone_inputs_with_overrides(base_inputs, wacc=w, terminal_growth=g)
                m = DCFModel(inp)
                price = m.implied_price(method=method)
            except ValueError:
                price = float("nan")
            row.append(price)
        data.append(row)
    df = pd.DataFrame(data, index=[f"{w:.1%}" for w in wacc_range], columns=[f"{g:.1%}" for g in growth_range])
    df.index.name = "WACC \\ g"
    return df


def sensitivity_wacc_exit_multiple(
    base_inputs: DCFInputs,
    wacc_range: list[float],
    multiple_range: list[float],
    method: str = "multiple",
) -> pd.DataFrame:
    """
    2-D sensitivity: WACC (rows) x Exit Multiple (columns).
    """
    data = []
    for w in wacc_range:
        row: list[float] = []
        for mult in multiple_range:
            inp = _clone_inputs_with_overrides(base_inputs, wacc=w, exit_multiple=mult)
            m = DCFModel(inp)
            price = m.implied_price(method=method)
            row.append(price)
        data.append(row)
    df = pd.DataFrame(data, index=[f"{w:.1%}" for w in wacc_range], columns=[f"{m:.1f}x" for m in multiple_range])
    df.index.name = "WACC \\ Multiple"
    return df


def scenario_comparison(
    base_assumptions: ModelAssumptions,
    scenarios: list[ScenarioAssumptions],
    wacc_map: dict[str, float] | None = None,
    terminal_growth_map: dict[str, float] | None = None,
    exit_multiple: float = 10.0,
    current_price: float | None = None,
) -> pd.DataFrame:
    """
    Run a full three-statement + DCF valuation under each scenario.

    Returns a comparison DataFrame with EV, Equity Value, Implied Price, Upside.
    """
    from src.financial_model.three_statement import ThreeStatementModel

    rows: list[dict] = []
    for scen in scenarios:
        assumptions = scen.apply_to(base_assumptions)
        wacc = (wacc_map or {}).get(scen.name, 0.09)
        tg = (terminal_growth_map or {}).get(scen.name, 0.02)
        # Allow scenario to carry wacc/terminal_growth directly
        if scen.wacc is not None:
            wacc = scen.wacc
        if scen.terminal_growth is not None:
            tg = scen.terminal_growth

        model = ThreeStatementModel(assumptions)
        dcf = DCFModel.from_three_statement(model, wacc=wacc, terminal_growth=tg, exit_multiple=exit_multiple)
        summ_g = dcf.summary("growth")
        summ_m = dcf.summary("multiple")

        row = {
            "Scenario": scen.name,
            "Revenue (Year 5)": assumptions.historical_revenue * np.prod([1 + g for g in assumptions.revenue_growth_list()]),
            "WACC": wacc,
            "Terminal Growth": tg,
            "EV (Growth)": summ_g["enterprise_value"],
            "Equity Value (Growth)": summ_g["equity_value"],
            "Implied Price (Growth)": summ_g["implied_share_price"],
            "EV (Multiple)": summ_m["enterprise_value"],
            "Equity Value (Multiple)": summ_m["equity_value"],
            "Implied Price (Multiple)": summ_m["implied_share_price"],
        }
        if current_price is not None:
            row["Upside (Growth)"] = (summ_g["implied_share_price"] / current_price - 1)
            row["Upside (Multiple)"] = (summ_m["implied_share_price"] / current_price - 1)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Scenario")
    return df
