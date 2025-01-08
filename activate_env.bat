@echo off
REM Set the name of your virtual environment folder
set VENV_DIR=virtual_env

REM Set your project information
set PROJECT_NAME=Primus-Anomaly-Detector
set PROJECT_DESCRIPTION=Primus sucks!
set PROJECT_AUTHOR=Created by Dmytro Riabchuk

REM Check if the virtual environment folder exists
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo --------------------------------------------------
    echo    Project: %PROJECT_NAME%
    echo    Description: %PROJECT_DESCRIPTION%
    echo    Author: %PROJECT_AUTHOR%
    echo --------------------------------------------------
    echo Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
    cmd /k
) else (
    echo Virtual environment "%VENV_DIR%" not found.
    echo Please ensure the virtual environment exists in the specified folder.
    pause
)
