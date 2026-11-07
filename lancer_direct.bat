@echo off
cd /d "%~dp0"
python -m pip install --quiet numpy pandas
python live_agent.py
pause
