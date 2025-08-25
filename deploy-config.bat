@echo off
setlocal enabledelayedexpansion

set "CONFIGS_DIR=configs"
set "BACKEND_DIR=."
set "SOURCE_DIR=source"

if "%~1"=="" (
    echo Usage: %0 [deployment^|local]
    pause
    exit /b 0
)

set "ENV=%~1"

if not "%ENV%"=="deployment" if not "%ENV%"=="local" (
    echo Error: Invalid environment '%ENV%'. Use 'deployment' or 'local'
    pause
    exit /b 1
)

if not exist "%CONFIGS_DIR%\%ENV%" (
    echo Error: Configuration directory '%CONFIGS_DIR%\%ENV%' not found
    pause
    exit /b 1
)

if not exist "%BACKEND_DIR%" (
    echo Error: Backend directory not found
    pause
    exit /b 1
)

if not exist "%SOURCE_DIR%" mkdir "%SOURCE_DIR%"

echo Deploying %ENV% configuration...

:: For local environment, remove deployment-specific files first
if "%ENV%"=="local" (
    if exist "nginx.conf" (
        del "nginx.conf"
        echo Removed nginx.conf
    )
    if exist "selfsigned.crt" (
        del "selfsigned.crt"
        echo Removed selfsigned.crt
    )
    if exist "selfsigned.key" (
        del "selfsigned.key"
        echo Removed selfsigned.key
    )
    if exist "start.sh" (
        del "start.sh"
        echo Removed start.sh
    )
)

:: Deploy .env to source directory
if exist "%CONFIGS_DIR%\%ENV%\.env" (
    copy "%CONFIGS_DIR%\%ENV%\.env" "%SOURCE_DIR%\.env" >nul
    echo Deployed .env to %SOURCE_DIR%
)

:: Deploy other files to current directory
for %%f in ("%CONFIGS_DIR%\%ENV%\*") do (
    if not "%%~nxf"==".env" (
        copy "%%f" "%%~nxf" >nul
        echo Deployed %%~nxf
    )
)

:: Show Docker commands for local environment
if "%ENV%"=="local" (
    echo.
    echo Docker commands to run:
    echo docker build -t sustainable-city-backend .
    echo docker run -p 8000:8000 -p 6379:6379 sustainable-city-backend
    echo.
    echo Or run locally with Python:
    echo python -m venv env
    echo env\Scripts\activate
    echo pip install -r requirements.txt
    echo python source\main.py
)

echo Configuration deployment completed
pause