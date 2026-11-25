@echo off
setlocal
cd /d "%~dp0"
echo Starting shared LAN studio. Use a LAN address printed below on your other devices.
call "%~dp0Start-Local.cmd" --host 0.0.0.0 %*
