#!/bin/bash

# Ensure the script runs from the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtual environment
source env/bin/activate

# Set PYTHONPATH to include both the root directory and the loraw subfolder
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/loraw"

# Run the Python script as if it is in the root folder
python loraw/run_gradio.py --ckpt-path ".\models\checkpoints\model.ckpt" --model-config ".\models\checkpoints\model_config.json" --lora-dir ".\models\loras"

# Keep the terminal open
read -p "Press [Enter] to continue..."