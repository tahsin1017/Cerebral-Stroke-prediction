#!/usr/bin/env bash
set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running training pipeline..."
python src/train.py

echo "Running interpretability and fairness audit..."
python src/audit.py

echo "✅ Reproduction complete! Check the figures/ folder."
