@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Enable-LAN-Firewall.ps1" %*
if errorlevel 1 echo Right-click this file and choose Run as administrator, then try again.
pause
