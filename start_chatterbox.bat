@echo off
title ChatterBox Web UI
cd /d "%~dp0"
echo Starting ChatterBox Web UI on http://localhost:7860 ...
start "ChatterBox Server" /B "%~dp0venv\Scripts\python.exe" "%~dp0app.py"
timeout /t 6 /nobreak >nul
start "" "http://localhost:7860"
echo.
echo ChatterBox is running. Close this window to stop the server.
pause
taskkill /F /IM python.exe /FI "WINDOWTITLE eq ChatterBox Server*" >nul 2>&1
