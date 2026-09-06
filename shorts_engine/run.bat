@echo off
setlocal EnableDelayedExpansion
title Batch Greek Shorts Processing Engine — Launcher

:: ============================================================
:: run.bat — 1-Click Windows Launcher
:: Responsibilities:
::   1. Verify Python 3.10+
::   2. Download FFmpeg static build if not on PATH
::   3. Create/reuse a .venv virtual environment
::   4. Install Python dependencies
::   5. Launch Streamlit
:: ============================================================

echo.
echo  ====================================================
echo   Batch Greek Shorts Processing Engine
echo   1-Click Launcher
echo  ====================================================
echo.

:: ── Step 1: Check Python ─────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  Ensure you check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Verify version is at least 3.10
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)
if !PYMAJ! LSS 3 (
    echo [ERROR] Python 3.10+ is required. Found: !PYVER!
    pause
    exit /b 1
)
if !PYMAJ! EQU 3 if !PYMIN! LSS 10 (
    echo [ERROR] Python 3.10+ is required. Found: !PYVER!
    pause
    exit /b 1
)
echo [OK] Python !PYVER! detected.

:: ── Step 2: Check / Download FFmpeg ──────────────────────────
set "BIN_DIR=%~dp0bin"
set "FFMPEG_BIN=%BIN_DIR%\ffmpeg.exe"

where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo [OK] FFmpeg already on PATH — skipping download.
    goto :setup_venv
)

if exist "%FFMPEG_BIN%" (
    echo [OK] Bundled FFmpeg found at %FFMPEG_BIN%.
    set "PATH=%BIN_DIR%;%PATH%"
    goto :setup_venv
)

echo [INFO] FFmpeg not found. Downloading static build from GitHub...
echo        ^(No admin rights required — extracted to .\bin\^)
echo.

:: Create bin dir
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

:: Use PowerShell to download the latest ffmpeg-master-latest-win64-gpl.zip
set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "FFMPEG_ZIP=%BIN_DIR%\ffmpeg.zip"
set "FFMPEG_EXTRACT=%BIN_DIR%\ffmpeg_extract"

powershell -NoProfile -Command ^
    "Write-Host 'Downloading FFmpeg (~80 MB)...'; " ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
    "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing; " ^
    "Write-Host 'Extracting...'; " ^
    "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%FFMPEG_EXTRACT%' -Force;"

if errorlevel 1 (
    echo [ERROR] FFmpeg download failed. Check your internet connection.
    pause
    exit /b 1
)

:: Move the ffmpeg/ffprobe executables directly into bin/
for /d %%d in ("%FFMPEG_EXTRACT%\*") do (
    if exist "%%d\bin\ffmpeg.exe" (
        copy /Y "%%d\bin\ffmpeg.exe" "%BIN_DIR%\ffmpeg.exe" >nul
        copy /Y "%%d\bin\ffprobe.exe" "%BIN_DIR%\ffprobe.exe" >nul
        copy /Y "%%d\bin\ffplay.exe" "%BIN_DIR%\ffplay.exe" >nul 2>&1
    )
)

:: Cleanup zip and extracted folder
del /Q "%FFMPEG_ZIP%" 2>nul
rmdir /S /Q "%FFMPEG_EXTRACT%" 2>nul

if not exist "%FFMPEG_BIN%" (
    echo [ERROR] FFmpeg extraction failed — executable not found at expected path.
    pause
    exit /b 1
)

set "PATH=%BIN_DIR%;%PATH%"
echo [OK] FFmpeg installed to %BIN_DIR%.

:: ── Step 3: Create virtual environment ───────────────────────
:setup_venv
set "VENV_DIR=%~dp0.venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: ── Step 4: Activate and install dependencies ─────────────────
call "%VENV_DIR%\Scripts\activate.bat"

echo [INFO] Installing / updating Python dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] pip install failed. See output above.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.

:: ── Step 5: Launch Streamlit ──────────────────────────────────
echo.
echo  Starting Streamlit... your browser will open automatically.
echo  Press Ctrl+C in this window to stop the server.
echo.

streamlit run "%~dp0app.py" --server.headless false --browser.gatherUsageStats false

endlocal
pause
