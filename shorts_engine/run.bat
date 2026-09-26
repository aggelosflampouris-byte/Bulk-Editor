@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: run.bat -- 1-Click Windows Launcher for Bulk Editor
:: Compatible: Windows 10 / 11
::
:: Steps:
::   1. Find Python 3.10+ (py launcher or python on PATH)
::   2. Find or download FFmpeg into .\bin\
::   3. Create / reuse .venv virtual environment
::   4. Install Python dependencies
::   5. Copy .env.example -> .env if not present
::   6. Launch Streamlit app
:: ============================================================

title Bulk Editor - Launcher

echo.
echo  ====================================================
echo    Batch Greek Shorts Processing Engine
echo    1-Click Launcher  (Windows)
echo  ====================================================
echo.

:: ── Resolve script directory (handles spaces in path) ─────────────────────────
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "REQ_FILE=%SCRIPT_DIR%\requirements.txt"
set "APP_FILE=%SCRIPT_DIR%\app.py"
set "BIN_DIR=%SCRIPT_DIR%\bin"
set "ENV_FILE=%SCRIPT_DIR%\.env"
set "ENV_EXAMPLE=%SCRIPT_DIR%\.env.example"

:: ── Step 1: Find Python 3.10+ ────────────────────────────────────────────────
echo [1/6] Checking Python version...

set "PYTHON_BIN="

:: Try Python Launcher (py.exe) first — most reliable on Windows 10/11
where py >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%V in ('py -3 --version 2^>^&1') do set "PYVER_LINE=%%V"
    set "PYTHON_BIN=py -3"
    goto :check_version
)

:: Fall back to python on PATH
where python >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%V in ('python --version 2^>^&1') do set "PYVER_LINE=%%V"
    set "PYTHON_BIN=python"
    goto :check_version
)

echo.
echo  [ERROR] Python is not installed or not on PATH.
echo.
echo  Please install Python 3.10+ from:
echo    https://www.python.org/downloads/windows/
echo.
echo  IMPORTANT: During installation, check the box:
echo    "Add Python to PATH"
echo.
pause
exit /b 1

:check_version
:: PYVER_LINE is like "Python 3.12.4"
for /f "tokens=2 delims= " %%V in ("!PYVER_LINE!") do set "PYVER=%%V"
for /f "tokens=1,2 delims=." %%A in ("!PYVER!") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)

if "!PY_MAJOR!"=="" (
    echo  [ERROR] Could not determine Python version from: !PYVER_LINE!
    pause
    exit /b 1
)

if !PY_MAJOR! LSS 3 (
    echo  [ERROR] Python 3.10 or newer is required. Found: !PYVER!
    pause
    exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo  [ERROR] Python 3.10 or newer is required. Found: !PYVER!
    echo  Please upgrade Python from https://www.python.org/downloads/windows/
    pause
    exit /b 1
)
echo  [OK] Python !PYVER! (!PYTHON_BIN!)

:: ── Step 2: Find or download FFmpeg ──────────────────────────────────────────
echo.
echo [2/6] Checking FFmpeg...

:: Check bundled bin\ first
if exist "%BIN_DIR%\ffmpeg.exe" (
    set "PATH=%BIN_DIR%;%PATH%"
    echo  [OK] Bundled FFmpeg found in bin\
    goto :setup_venv
)

:: Check system PATH
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo  [OK] System FFmpeg found on PATH.
    goto :setup_venv
)

echo  [INFO] FFmpeg not found. Downloading static build (~80 MB)...
echo         (No admin rights required -- saved to .\bin\)
echo.

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

:: Download and extract via PowerShell (available on all Windows 10+)
set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "FFMPEG_ZIP=%BIN_DIR%\ffmpeg_dl.zip"
set "FFMPEG_EXTRACT=%BIN_DIR%\ffmpeg_extract"

:: Single-line PowerShell to avoid ^ continuation issues
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol='Tls12,Tls13'; Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing; Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%FFMPEG_EXTRACT%' -Force"
if errorlevel 1 (
    echo.
    echo  [ERROR] FFmpeg download failed. Check internet connection or install manually:
    echo    https://www.gyan.dev/ffmpeg/builds/
    echo  Then place ffmpeg.exe and ffprobe.exe inside: %BIN_DIR%
    echo.
    if exist "%FFMPEG_ZIP%" del /Q "%FFMPEG_ZIP%"
    pause
    exit /b 1
)

:: Locate and copy the executables out of the versioned subfolder
for /d %%D in ("%FFMPEG_EXTRACT%\*") do (
    if exist "%%D\bin\ffmpeg.exe" (
        copy /Y "%%D\bin\ffmpeg.exe"  "%BIN_DIR%\ffmpeg.exe"  >nul
        copy /Y "%%D\bin\ffprobe.exe" "%BIN_DIR%\ffprobe.exe" >nul
        copy /Y "%%D\bin\ffplay.exe"  "%BIN_DIR%\ffplay.exe"  >nul 2>&1
    )
)

:: Cleanup
del /Q "%FFMPEG_ZIP%" 2>nul
rmdir /S /Q "%FFMPEG_EXTRACT%" 2>nul

if not exist "%BIN_DIR%\ffmpeg.exe" (
    echo  [ERROR] FFmpeg extraction failed -- executable not found.
    pause
    exit /b 1
)

set "PATH=%BIN_DIR%;%PATH%"
echo  [OK] FFmpeg installed to bin\

:: ── Step 3: Create virtual environment ───────────────────────────────────────
:setup_venv
echo.
echo [3/6] Setting up Python virtual environment...

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo  [OK] Virtual environment already exists.
    goto :install_deps
)

%PYTHON_BIN% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo.
    echo  [ERROR] Failed to create virtual environment.
    echo  This may happen if python3-venv is not installed.
    echo  Try: pip install virtualenv  OR  reinstall Python with pip enabled.
    echo.
    pause
    exit /b 1
)
echo  [OK] Virtual environment created.

:: ── Step 4: Install dependencies ─────────────────────────────────────────────
:install_deps
echo.
echo [4/6] Installing Python dependencies (first run may take a few minutes)...

"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo  [WARNING] Could not upgrade pip -- continuing anyway.
)

"%VENV_DIR%\Scripts\pip.exe" install -r "%REQ_FILE%"
if errorlevel 1 (
    echo.
    echo  [ERROR] Failed to install dependencies.
    echo  Check your internet connection and the output above for details.
    echo.
    pause
    exit /b 1
)
echo  [OK] Dependencies installed.

:: ── Step 5: Ensure .env exists ───────────────────────────────────────────────
echo.
echo [5/6] Checking configuration...

if not exist "%ENV_FILE%" (
    if exist "%ENV_EXAMPLE%" (
        copy "%ENV_EXAMPLE%" "%ENV_FILE%" >nul
        echo  [OK] Created .env from .env.example -- please add your API keys.
    ) else (
        echo  [INFO] No .env.example found; skipping .env creation.
    )
) else (
    echo  [OK] .env file found.
)

:: ── Step 6: Launch Streamlit ──────────────────────────────────────────────────
echo.
echo [6/6] Launching Streamlit...
echo.
echo  ====================================================
echo    App is starting! Your browser will open shortly.
echo    To stop the server, press Ctrl+C in this window.
echo  ====================================================
echo.

:: Set PYTHONPATH so both the engine root and parent are importable
set "PYTHONPATH=%SCRIPT_DIR%;%SCRIPT_DIR%\.."
if defined PYTHONPATH_ORIG set "PYTHONPATH=%PYTHONPATH%;%PYTHONPATH_ORIG%"

"%VENV_DIR%\Scripts\streamlit.exe" run "%APP_FILE%" ^
    --server.headless false ^
    --browser.gatherUsageStats false

:: ── Keep window open on unexpected exit ──────────────────────────────────────
echo.
echo  Streamlit has stopped.
echo.
pause
endlocal
