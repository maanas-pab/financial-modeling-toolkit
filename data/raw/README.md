# Raw Data

Immutable source files. Do not edit in place — transformations go to `data/processed/`.

| File | Description |
|---|---|
| `sample_assumptions.csv` | Long-format assumptions for the three-statement model (key/value). Synthetic demonstration dataset — not actual company financials. |
| `synthetic_prices.csv` | Daily synthetic price panel (2018-01-01 → ~2022, 1260 bars, 5 assets: AAPL/MSFT/JPM/XOM/TLT). Generated with `numpy` lognormal-like returns, fixed seed `42`. **Synthetic demonstration dataset — not actual company financials.** |

Regenerate synthetic prices:

```python
python -c "import pandas as pd, numpy as np; ..."
# or simply run python examples/run_full_demo.py which uses the same generator
```
