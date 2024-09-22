@echo off
REM Lora training launch

REM Activate the virtual environment
call env\Scripts\activate

REM Set PYTHONPATH to the root directory to include the loraw subfolder
set PYTHONPATH=%CD%;%CD%\loraw

python loraw\training_setup.py
pause