@echo off
REM One-click runner for document-level pipeline (assumes MCC training already done)
SETLOCAL ENABLEDELAYEDEXPANSION

REM Move to repository root (script location)
cd /d %~dp0

REM Change into src
cd src || (
  echo Failed to change directory to src
  exit /b 1
)

echo Checking for Python...
python --version >nul 2>&1 || (
  echo Python not found on PATH. Please install Python 3.10+ and add it to PATH.
  exit /b 1
)

REM Create venv if missing
if not exist .venv\Scripts\activate.bat (
  echo Creating virtual environment .venv
  python -m venv .venv
  if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment
    exit /b 1
  )
)

echo Activating virtual environment
call .venv\Scripts\activate.bat || (
  echo Failed to activate virtual environment
  exit /b 1
)

echo Skipping pip upgrade step as requested.

REM Install dependencies if requirements.txt exists at repo root; else install minimal set
if exist ..\requirements.txt (
  echo Installing from requirements.txt
  pip install -r ..\requirements.txt
  if %ERRORLEVEL% NEQ 0 (
    echo Failed to install from requirements.txt
    exit /b 1
  )
) else (
  echo Installing minimal dependencies (this may take a while)...
  pip install transformers tqdm numpy scikit-learn pandas beautifulsoup4 requests
  if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies
    exit /b 1
  )
)

echo Preparing news dataset...
python -m scripts.prepare_news_dataset || (
  echo prepare_news_dataset failed. See message above.
  exit /b 1
)

echo Running document-level classification pipeline...
python -m scripts.run_doc_classification --news-train data/news_train.jsonl --news-test data/news_test.jsonl --mcc-checkpoint checkpoints/mcc_bert.pt
if %ERRORLEVEL% NEQ 0 (
  echo Document classification failed.
  exit /b %ERRORLEVEL%
)

echo Pipeline completed successfully.
endlocal
exit /b 0
