#!/bin/bash
# How to run the authors' code (S4c agent output): a fresh python:3.11-slim container reproduces the example.
set -euo pipefail
python -m venv /opt/env && . /opt/env/bin/activate
pip install --no-cache-dir numpy==2.1.0 scipy==1.13.1 tifffile==2024.8.30
python fit_orientations.py --example     # prints the fitted orientation of the two example frames
