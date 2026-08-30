"""Data loading utilities — CSV, Excel, DataFrame passthrough."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.financial_model.assumptions import ModelAssumptions


def load_assumptions_from_csv(path: str | Path, **kwargs) -> ModelAssumptions:
    """
    Load assumptions from a CSV file.

    Expected format: two columns 'parameter' and 'value' or a single-row
    wide table with parameter names as columns.  List parameters can be
    stored as comma-separated strings (e.g., "0.06,0.05,0.04").
    """
    path = Path(path)
    df = pd.read_csv(path)
    # Try key-value long format
    if "parameter" in df.columns and "value" in df.columns:
        raw = dict(zip(df["parameter"], df["value"]))
    else:
        # Wide single row
        raw = df.iloc[0].to_dict()

    parsed = _parse_assumption_dict(raw)
    return ModelAssumptions(**parsed)


def load_assumptions_from_excel(path: str | Path, sheet_name: str = "Assumptions") -> ModelAssumptions:
    """
    Load assumptions from an Excel sheet.

    Same conventions as CSV.
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if "parameter" in df.columns and "value" in df.columns:
        raw = dict(zip(df["parameter"], df["value"]))
    else:
        raw = df.iloc[0].to_dict()
    parsed = _parse_assumption_dict(raw)
    return ModelAssumptions(**parsed)


def _parse_assumption_dict(raw: dict) -> dict:
    """Parse string representations into appropriate Python types."""
    # Keys that may be lists (only treat as list if comma-separated)
    list_keys = {"revenue_growth", "gross_margin", "opex_margin", "da_pct_revenue", "capex_pct_revenue", "debt_issuance", "debt_repayment"}
    out: dict = {}
    for k, v in raw.items():
        if pd.isna(v):
            continue
        k = str(k).strip()
        if k in list_keys and isinstance(v, str) and "," in v:
            # "0.06, 0.05,0.04" -> list
            out[k] = [float(x.strip()) for x in v.split(",") if x.strip() != ""]
        else:
            # Try numeric scalar
            try:
                out[k] = float(v)
                # Keep int for forecast_years etc
                if k in ("forecast_years", "base_year_label"):
                    out[k] = int(float(v))
            except (ValueError, TypeError):
                # For list_keys with single string numeric like "0.40" -> treat as scalar float
                if k in list_keys and isinstance(v, str):
                    try:
                        out[k] = float(v.strip())
                    except ValueError:
                        out[k] = v
                else:
                    out[k] = v
    return out


def export_results_to_excel(
    path: str | Path,
    income_statement: pd.DataFrame | None = None,
    balance_sheet: pd.DataFrame | None = None,
    cash_flow: pd.DataFrame | None = None,
    dcf_summary: dict | None = None,
) -> None:
    """Write selected outputs back to Excel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if income_statement is not None:
            income_statement.to_excel(writer, sheet_name="Income Statement")
        if balance_sheet is not None:
            balance_sheet.to_excel(writer, sheet_name="Balance Sheet")
        if cash_flow is not None:
            cash_flow.to_excel(writer, sheet_name="Cash Flow")
        if dcf_summary is not None:
            pd.Series(dcf_summary).to_frame("Value").to_excel(writer, sheet_name="DCF Summary")
