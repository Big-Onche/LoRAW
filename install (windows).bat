@echo off
REM Windows install script

REM Ensure the script runs from the directory where the batch file is located
cd /d %~dp0

echo Creating virtual environment...
python -m venv env

REM Activate the virtual environment
call .\env\Scripts\activate

REM install wheel (fixes some deprecation warnings with legacy 'setup.py install method)
pip install wheel

REM Navigate to the folder containing setup.py
cd loraw

REM Install torch
pip install torch==2.4.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Install the package
pip install .

REM Go back to the original directory
cd ..

echo Installation complete.
pause