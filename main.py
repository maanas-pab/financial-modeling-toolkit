#!/usr/bin/env python3
"""Root wrapper — allows `python main.py` as alternative to `python -m src.main`."""
from src.main import main

if __name__ == "__main__":
    raise SystemExit(main())
