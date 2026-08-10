@echo off
echo.
echo ========================================
echo   AgriSathi AI Advisor - Quick Start
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
pip install -r backend\requirements.txt --quiet

echo [3/3] Starting backend server...
echo.
echo  Open in browser:
echo    http://127.0.0.1:8000/docs
echo    frontend\dashboard\index.html
echo.
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
pause
