@echo off
echo Starting MTT Radar Engine...
start "MTT Server" cmd /c "python -m server.app"
timeout /t 2 >nul
start http://127.0.0.1:8000
