#!/bin/bash

# Ensure the script runs from the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating virtual environment..."
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# install wheel (fixes some deprecation warnings with legacy 'setup.py install method)
pip install wheel

# Navigate to the folder containing setup.py
cd loraw

# Install torch
pip install torch==2.4.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the package
pip install .

# Go back to the original directory
cd ..

echo "Installation complete."