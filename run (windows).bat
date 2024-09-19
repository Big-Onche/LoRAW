@echo off
REM LoRAW gradio launch

REM Activate the virtual environment
call env\Scripts\activate

REM Set PYTHONPATH to the root directory to include the loraw subfolder
set PYTHONPATH=%CD%;%CD%\loraw

REM Run the Python script as if it is in the root folder
python loraw\run_gradio.py --ckpt-path ".\models\checkpoints\model.ckpt" --model-config ".\models\checkpoints\model_config.json" --lora-dir ".\models\loras"

REM Pause to keep the terminal open
pause