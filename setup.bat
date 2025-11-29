@echo off
echo 🚀 Fine-FinTech: Gemma3 Fine-tuning Setup
echo ========================================

echo.
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Setting up project environment...
python scripts\setup.py

echo.
echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Edit config\training_config.yaml if needed
echo 2. Run: python src\finetune.py
echo.
pause