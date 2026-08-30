#!/usr/bin/env python3
"""Legacy shim for `pip install -e .` on older pip (<22) that does not support PEP 660 editable with pyproject.toml alone."""
from setuptools import setup

setup()
