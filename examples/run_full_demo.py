#!/usr/bin/env python3
"""
Full demonstration of the Financial Modeling & Quantitative Finance Toolkit.

When executed:
    python examples/run_full_demo.py

It:
 1. Loads the example dataset (synthetic — labelled as such)
 2. Builds the three-statement financial model
 3. Forecasts statements
 4. Runs WACC + DCF (both Gordon and exit-multiple)
 5. Generates sensitivity analysis (WACC×g, WACC×Multiple)
 6. Runs Bull/Base/Bear scenarios
 7. Loads portfolio price data (synthetic)
 8. Calculates portfolio statistics
 9. Optimizes portfolios (min-var, max-Sharpe, target)
10. Generates the efficient frontier
11. Runs Monte Carlo simulation
12. Generates charts
13. Exports all outputs to outputs/

This is the canonical end-to-end example referenced by the CLI `--demo`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import cmd_full_demo

if __name__ == "__main__":
    # Also allow import without side effects; only run when executed directly.
    raise SystemExit(cmd_full_demo())
