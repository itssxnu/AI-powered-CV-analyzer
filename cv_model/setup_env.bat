@echo off
echo ==========================================
echo Starting PyCharm Virtual Environment Setup
echo ==========================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not added to PATH. Please install Python first.
    pause
    exit /b
)

if not exist ".venv" (
    echo Creating virtual environment '.venv'...
    python -m venv .venv
) else (
    echo Virtual environment '.venv' already exists.
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo ==========================================
echo Setup complete! Virtual environment is ready.
echo To activate the virtual environment manually, run:
echo .venv\Scripts\activate
echo ==========================================
pause
