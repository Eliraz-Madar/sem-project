@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ========================================
echo Running Complete Pipeline
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
echo Step 0: Setting up virtual environment
echo ========================================
echo.

REM Create virtual environment if it doesn't exist
if not exist .venv\ (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

echo.
echo Installing dependencies...
if exist requirements.txt (
    echo Installing from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies from requirements.txt.
        exit /b 1
    )
) else (
    echo requirements.txt not found, installing minimal dependencies...
    python -m pip install transformers torch scikit-learn numpy tqdm
    if errorlevel 1 (
        echo [ERROR] Failed to install minimal dependencies.
        exit /b 1
    )
)

echo.
echo ========================================
echo Step 1: Converting Webis corpus to JSONL
echo ========================================
echo.

cd src
if errorlevel 1 (
    echo [ERROR] Failed to change to src directory.
    exit /b 1
)

python -m scripts.convert_webis_to_jsonl
if errorlevel 1 (
    echo [ERROR] Step 1 failed: convert_webis_to_jsonl
    exit /b 1
)

echo [OK] Step 1 completed successfully.

echo.
echo ========================================
echo Step 2: Training MCC model
echo ========================================
echo.

python -m scripts.run_mcc_training --train data/editorials.jsonl --dev data/editorials.jsonl
if errorlevel 1 (
    echo [ERROR] Step 2 failed: run_mcc_training
    exit /b 1
)

echo [OK] Step 2 completed successfully.

echo.
echo ========================================
echo Step 3: Preparing news dataset
echo ========================================
echo.

python -m scripts.prepare_news_dataset
if errorlevel 1 (
    echo [ERROR] Step 3 failed: prepare_news_dataset
    exit /b 1
)

echo [OK] Step 3 completed successfully.

echo.
echo ========================================
echo Step 4: Running document classification
echo ========================================
echo.

python -m scripts.run_doc_classification --news-train data/news_train.jsonl --news-test data/news_test.jsonl --mcc-checkpoint checkpoints/mcc_bert.pt
if errorlevel 1 (
    echo [ERROR] Step 4 failed: run_doc_classification
    exit /b 1
)

echo [OK] Step 4 completed successfully.

echo.
echo ========================================
echo Pipeline completed successfully!
echo ========================================

cd ..
exit /b 0
