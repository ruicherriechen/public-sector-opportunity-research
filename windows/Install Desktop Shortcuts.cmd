@echo off
setlocal
set "SETUP_SCRIPT=%~dp0Install Desktop Shortcuts.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SETUP_SCRIPT%"
if errorlevel 1 (
  echo.
  echo Shortcut setup did not finish. Review the message above, then press any key.
  pause >nul
  exit /b 1
)
echo.
echo Shortcut setup is complete. Press any key to close this window.
pause >nul
exit /b 0
