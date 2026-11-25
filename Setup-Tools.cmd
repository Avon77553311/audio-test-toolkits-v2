@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Setup-Tools.ps1" %*
if errorlevel 1 (
  echo Setup failed. See the error above, then run this file again.
  pause
  exit /b 1
)
echo Setup complete. Run Start-Local.cmd to open the website.
pause
