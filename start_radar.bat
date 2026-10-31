@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Starting MTT Radar Engine...
echo Folder: %CD%
echo.

REM Already running? Just open the radar page.
powershell -NoProfile -Command "try { $ProgressPreference='SilentlyContinue'; $null = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/' -TimeoutSec 2; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  echo The radar server is already running.
  start "" "http://127.0.0.1:8000/"
  echo Opened http://127.0.0.1:8000/
  timeout /t 3 >nul
  exit /b 0
)

call :findpython
if not defined PYEXE (
  echo Could not find a working Python 3.
  echo.
  echo Install Python 3 from https://www.python.org/downloads/
  echo During setup, tick "Add python.exe to PATH".
  echo The Microsoft Store "python" shortcut does not run this project.
  echo.
  pause
  exit /b 1
)

echo Using Python:
echo   %PYEXE%
echo.

"%PYEXE%" -c "import numpy, scipy" >nul 2>&1
if errorlevel 1 (
  echo Installing numpy and scipy. This can take a minute the first time...
  echo.
  "%PYEXE%" -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo.
    echo Package install failed. Open Command Prompt in this folder and run:
    echo   "%PYEXE%" -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
  )
)

echo.
echo Starting the server. Leave the "MTT Radar Server" window open while you use the radar.
start "MTT Radar Server" /D "%~dp0" cmd /k ""%PYEXE%" -m server.app"

echo Waiting for http://127.0.0.1:8000 ...
set /a N=0
:waitloop
set /a N+=1
powershell -NoProfile -Command "try { $ProgressPreference='SilentlyContinue'; $null = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/' -TimeoutSec 2; exit 0 } catch { exit 1 }"
if not errorlevel 1 goto :ready
if %N% GEQ 45 goto :notready
timeout /t 1 >nul
goto :waitloop

:ready
start "" "http://127.0.0.1:8000/"
echo.
echo Radar is open in your browser.
echo Close the "MTT Radar Server" window when you are finished.
timeout /t 5 >nul
exit /b 0

:notready
echo.
echo The server did not start. Read the error in the "MTT Radar Server" window.
echo Common causes: port 8000 is already taken, or numpy/scipy failed to install.
echo.
pause
exit /b 1

:findpython
set "PYEXE="
where py >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%E in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%E"
  )
)
if defined PYEXE exit /b 0

for /f "delims=" %%P in ('where python 2^>nul') do (
  if not defined PYEXE call :trypy "%%P"
)
if defined PYEXE exit /b 0

for /f "delims=" %%P in ('where python3 2^>nul') do (
  if not defined PYEXE call :trypy "%%P"
)
exit /b 0

:trypy
echo %~1 | findstr /i /c:"WindowsApps" >nul
if not errorlevel 1 exit /b 0
for /f "delims=" %%E in ('"%~1" -c "import sys; print(sys.executable)" 2^>nul') do (
  if not defined PYEXE set "PYEXE=%%E"
)
exit /b 0
