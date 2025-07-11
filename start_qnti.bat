@echo off
echo Starting Quantum Nexus Trading Intelligence...
echo.

REM Set UTF-8 encoding
chcp 65001 >nul 2>&1

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "qnti_main_system.py" (
    echo ERROR: qnti_main_system.py not found.
    echo Please ensure all QNTI files are in the current directory.
    pause
    exit /b 1
)

echo Starting QNTI in safe mode (no auto-trading)...
echo Press Ctrl+C to stop the system
echo.

python qnti_main_system.py --no-auto-trading --debug

echo.
echo QNTI has stopped.
pause
