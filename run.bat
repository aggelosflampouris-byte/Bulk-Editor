@echo off
setlocal
title Batch Greek Shorts Processing Engine — Launcher

echo.
echo ====================================================
echo   Batch Greek Shorts Processing Engine
echo   Launching from root directory...
echo ====================================================
echo.

if not exist "%~dp0shorts_engine\run.bat" (
    echo [ERROR] Could not find shorts_engine\run.bat.
    echo Please make sure the shorts_engine folder exists in the project root.
    pause
    exit /b 1
)

cd /d "%~dp0shorts_engine"
call run.bat %*
endlocal
