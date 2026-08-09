@echo off
title WALP
cd /d "%~dp0"

rem This machine has more than one Python. Only the one with the libraries
rem installed can run the game, so pick whichever actually imports pygame
rem instead of trusting whatever "python" happens to point at.
set "PYEXE="
for %%C in (python3.11.exe python3.12.exe python3.10.exe python.exe) do (
    if not defined PYEXE (
        %%C -c "import pygame, cv2, mediapipe, numpy" >nul 2>&1 && set "PYEXE=%%C"
    )
)

if not defined PYEXE (
    echo.
    echo ---------------------------------------------------------------
    echo  No Python found with the required libraries installed.
    echo.
    echo  Install them with:
    echo     python -m pip install pygame opencv-python mediapipe numpy
    echo.
    echo  Note: mediapipe does not support Python 3.13 yet - use 3.11.
    echo ---------------------------------------------------------------
    echo.
    pause
    exit /b 1
)

echo Using %PYEXE%
%PYEXE% walp2.py %*

if errorlevel 1 (
    echo.
    echo WALP stopped with an error. Press a key to close.
    pause >nul
)
