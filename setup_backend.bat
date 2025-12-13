@echo off
echo Setting up Python backend environment...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Setup complete!
echo.
echo To start the backend server, run:
echo   cd backend
echo   python app.py
echo.
pause
