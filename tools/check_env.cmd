@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ========================================
echo Environment Check
echo ========================================
echo.

REM Check Python availability
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not available in PATH.
    echo Please install Python 3.10+ and ensure it's in your PATH.
    exit /b 1
)

echo Python version:
python --version

echo.
echo Python location:
where python

echo.
echo Pip version:
python -m pip --version

echo.
REM Check if src\ directory exists
if not exist src\ (
    echo [ERROR] src\ directory not found.
    echo Please run this script from the repository root.
    exit /b 1
)

echo [OK] src\ directory found.
echo.
echo ========================================
echo Environment check passed!
echo ========================================

exit /b 0
