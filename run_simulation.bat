@echo off
echo ========================================
echo QNTI Trading System - Full Simulation
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Install required packages
echo Installing required packages...
pip install requests websocket-client Pillow selenium

:: Check if QNTI server is running
echo Checking QNTI server status...
curl -s http://localhost:5000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: QNTI server may not be running at http://localhost:5000
    echo Please start the QNTI server first
    echo.
    set /p choice="Continue anyway? (y/n): "
    if /i not "%choice%"=="y" (
        echo Simulation cancelled
        pause
        exit /b 1
    )
)

echo.
echo Starting QNTI Full System Simulation...
echo.

:: Run the full simulation
python run_qnti_full_simulation.py

echo.
echo Simulation completed!
echo Check the generated report files for detailed results.
echo.
pause 