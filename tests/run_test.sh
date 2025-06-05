#!/bin/bash

uv run tests/test_imports.py
echo "Running algorithm tests..."
uv run python -m unittest discover -s tests/algorithms/genesis_algorithms/ -p "test_*.py"
