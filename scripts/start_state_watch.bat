@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%\src"
if exist "%CD%\.venv\Scripts\python.exe" (
  set "PY=%CD%\.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
echo Restarting mm-assist status bar...
"%PY%" -m mm_assist watch --restart --detach --interval 0.5
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo Failed, code %RC%
  pause
)
exit /b %RC%
