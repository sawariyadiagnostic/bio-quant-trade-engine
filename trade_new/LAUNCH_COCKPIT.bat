@echo off
setlocal enabledelayedexpansion
title BIO-QUANT ALPHA COCKPIT LAUNCHER

echo =====================================================================
echo        BIO-QUANT ALPHA-OS: SOVEREIGN COCKPIT LAUNCHER
echo =====================================================================
echo.

:: 1. Determine base directory and normalize to trade_new
set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

if exist "%BASE_DIR%trade_new\src\ui\server.py" (
    cd /d "%BASE_DIR%trade_new"
) else if not exist "src\ui\server.py" (
    echo [ERROR] Could not locate src\ui\server.py from %CD%
    pause
    exit /b 1
)

:: 2. Locate Python executable (VirtualEnv > Local AppData > System PATH)
set "PY_CMD="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0trade_new\.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0trade_new\.venv\Scripts\python.exe"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python 3.11+ was not detected on this machine.
    echo Please install Python or ensure it is added to your system PATH.
    pause
    exit /b 1
)

echo [INFO] Python Executable : %PY_CMD%
echo [INFO] Active Directory  : %CD%
echo [INFO] Target Cockpit URL: http://localhost:7860
echo.

:: 3. Check if server is already running on Port 7860
netstat -ano | findstr ":7860" | findstr "LISTENING" >nul 2>&1
if !errorlevel! equ 0 (
    echo [INFO] Cockpit server is ALREADY ACTIVE on Port 7860.
    echo [INFO] Launching browser directly...
    start http://localhost:7860
    ping 127.0.0.1 -n 2 >nul
    exit /b 0
)

:: 4. Pre-launch background browser opener (fires after 2-second server warmup)
echo [INFO] Starting Cockpit Server...
echo [INFO] Your default web browser will open http://localhost:7860 automatically.
echo [INFO] To stop the server at any time, close this window or press Ctrl+C.
echo.

start /b cmd /c "ping 127.0.0.1 -n 3 >nul & start http://localhost:7860"

:: 5. Launch FastAPI Uvicorn Server in foreground
"%PY_CMD%" -m uvicorn src.ui.server:app --host 127.0.0.1 --port 7860 --log-level info

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Cockpit server exited with error code !errorlevel!.
    pause
)
