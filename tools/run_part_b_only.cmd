@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ========================================
echo Running Document Classification Only
echo ========================================
echo.

REM Check environment first
call "%~dp0check_env.cmd"
if errorlevel 1 (
    echo [ERROR] Environment check failed.
    exit /b 1
)

echo.
echo ========================================
echo Setting up virtual environment
echo ========================================
echo.

REM Check if virtual environment exists
if not exist .venv\ (
    echo [ERROR] Virtual environment not found.
    echo Please run tools\run_all.cmd first to set up the environment.
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo.
echo Changing to src directory...
cd src
if errorlevel 1 (
    echo [ERROR] Failed to change to src directory.
    exit /b 1
)

echo.
echo Checking for MCC checkpoint...
if not exist checkpoints\mcc_bert.pt (
    echo [ERROR] MCC checkpoint not found at checkpoints\mcc_bert.pt
    echo Please train the MCC model first by running tools\run_all.cmd
    echo or provide a valid checkpoint path.
    exit /b 1
)

echo [OK] MCC checkpoint found.

echo.
echo ========================================
echo Running document classification
echo ========================================
echo.

python -m scripts.run_doc_classification --news-train data/news_train.jsonl --news-test data/news_test.jsonl --mcc-checkpoint checkpoints/mcc_bert.pt
if errorlevel 1 (
    echo [ERROR] Document classification failed.
    exit /b 1
)

echo.
echo ========================================
echo Document classification completed!
echo ========================================

cd ..
exit /b 0
