@echo off
:: Setup script for creating a virtual environment for the Wikipedia 3D Graph Visualizer on Windows

:: Set the name of the virtual environment
set VENV_NAME=.venv
set REQUIREMENTS_FILE=requirements.txt

echo Setting up virtual environment for Wikipedia 3D Graph Visualizer...

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is required but not found. Please install Python and try again.
    exit /b 1
)

:: Create the virtual environment
echo Creating virtual environment at %VENV_NAME%...
python -m venv %VENV_NAME%

:: Check if venv creation was successful
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo Failed to create virtual environment. Please check your Python installation.
    exit /b 1
)

:: Activate the virtual environment
echo Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies from %REQUIREMENTS_FILE%...
pip install --upgrade pip
pip install -r %REQUIREMENTS_FILE%

:: Verify installation
if %ERRORLEVEL% equ 0 (
    echo Virtual environment setup complete!
    echo.
    echo To activate the virtual environment, run:
    echo   %VENV_NAME%\Scripts\activate.bat
    echo.
    echo To start the application, run:
    echo   python main.py
    echo.
    echo To generate a sample dataset, run:
    echo   python generate_sample.py
    echo.
    echo To deactivate the virtual environment when done, run:
    echo   deactivate
) else (
    echo Failed to install dependencies.
    exit /b 1
)
