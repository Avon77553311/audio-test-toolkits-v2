@echo off
setlocal
cd /d "%~dp0"
set "STUDIO_JAVA=java"
if defined JAVA_HOME if exist "%JAVA_HOME%\bin\java.exe" set "STUDIO_JAVA=%JAVA_HOME%\bin\java.exe"
"%STUDIO_JAVA%" -version >nul 2>&1
if errorlevel 1 (
  echo Java was not found. Install Java 17 or later and try again.
  pause
  exit /b 1
)
if not exist "%~dp0server\audio-studio-server.jar" (
  echo The server JAR is missing. Extract the complete local deployment package.
  pause
  exit /b 1
)
echo Starting Audio Toolkit Short. Open http://127.0.0.1:8081/ in your browser.
echo To use another port: Start-Local.cmd --port 8082
"%STUDIO_JAVA%" -jar "%~dp0server\audio-studio-server.jar" --root "%~dp0dist" %*
if errorlevel 1 pause
