@echo off
REM LoRAW gradio launch

REM Activate the virtual environment
call env\Scripts\activate

REM Set PYTHONPATH to the root directory to include the loraw subfolder
set PYTHONPATH=%CD%;%CD%\loraw

REM Run the Python script as if it is in the root folder
python loraw\gui.py

REM Pause to keep the terminal open
pause