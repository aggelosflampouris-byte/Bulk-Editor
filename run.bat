@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: run.bat -- Root launcher for Bulk Editor (Windows 10/11)
:: Delegates to shorts_engine\run.bat with the correct
:: working directory and PYTHONPATH pre-configured.
:: ============================================================

title Bulk Editor - Launcher

echo.
echo  ====================================================
echo    Batch Greek Shorts Processing Engine
echo  ====================================================
echo.

:: ── Resolve this script's directory (handles spaces in paths) ─────────────────
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "ENGINE_BAT=%ROOT_DIR%\shorts_engine\run.bat"

if not exist "%ENGINE_BAT%" (
    echo  [ERROR] Could not find shorts_engine\run.bat
    echo.
    echo  Make sure the "shorts_engine" folder exists next to this run.bat file.
    echo  If you downloaded the zip, re-extract it and try again.
    echo.
    pause
    exit /b 1
)

set "PYTHONPATH_PREV=%PYTHONPATH%"
set "PYTHONPATH=%ROOT_DIR%;%ROOT_DIR%\shorts_engine"
if defined PYTHONPATH_PREV set "PYTHONPATH=%PYTHONPATH%;%PYTHONPATH_PREV%"

:: Hand off to the engine launcher (which manages venv, ffmpeg, deps, and launch)
call "%ENGINE_BAT%" %*

:: Propagate exit code
endlocal
exit /b %ERRORLEVEL%
