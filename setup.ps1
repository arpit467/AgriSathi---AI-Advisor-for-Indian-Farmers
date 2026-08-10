# AgriSathi — Windows Setup Script
# Run this in PowerShell from the AgriSathi folder:
# PowerShell -ExecutionPolicy Bypass -File setup.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  AgriSathi AI Advisor — Windows Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found! Install from https://python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv
Write-Host "[OK] venv created" -ForegroundColor Green

# Activate venv
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# Install requirements
Write-Host "[3/4] Installing requirements..." -ForegroundColor Cyan
pip install -r backend\requirements.txt --quiet
Write-Host "[OK] Requirements installed" -ForegroundColor Green

# Run backend
Write-Host "[4/4] Starting FastAPI backend..." -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  Backend running at:" -ForegroundColor Yellow
Write-Host "  http://127.0.0.1:8000" -ForegroundColor Yellow  
Write-Host "  http://127.0.0.1:8000/docs  (API Docs)" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Also open: frontend/dashboard/index.html" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

Set-Location backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
