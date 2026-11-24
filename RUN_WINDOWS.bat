@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python train.py --dataset Twitter15 --epochs 5 --limit-per-class 50 --batch-size 8
pause
