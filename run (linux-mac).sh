#!/bin/bash

# Ensure the script runs from the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtual environment
source env/bin/activate

# Set PYTHONPATH to include both the root directory and the loraw subfolder
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/loraw"

# Run the Python script as if it is in the root folder
python loraw\gui.py

# Keep the terminal open
read -p "Press [Enter] to continue..."