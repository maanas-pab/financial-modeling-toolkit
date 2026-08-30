# Processed Data

Derived, analysis-ready datasets.

| File | Description |
|---|---|
| `sample_prices.csv` | First few rows of `synthetic_prices.csv` for quick preview. |

Full processed artefacts (returns, covariance, etc.) are generated on the fly by `src/portfolio/` and cached only if explicitly exported via `examples/run_full_demo.py` to `outputs/`.
