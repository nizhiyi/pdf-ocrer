@echo off
rem ============================================================
rem  pdf-ocrer one-click installer build (double-click to run).
rem  Runs packaging\build.ps1 with PowerShell 7+:
rem    1) creates the paddle-free build venv (first run: ~2-3 min)
rem    2) freezes the app with PyInstaller
rem    3) compiles the installer with Inno Setup
rem  Output: packaging\Output\pdf-ocrer-setup-<version>.exe
rem  Requirements: PowerShell 7 (pwsh) + Inno Setup 6 installed.
rem ============================================================
setlocal
cd /d "%~dp0"

set "PWSH="
where pwsh.exe >nul 2>nul && set "PWSH=pwsh.exe"
if not defined PWSH if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" set "PWSH=%ProgramFiles%\PowerShell\7\pwsh.exe"
if not defined PWSH if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\pwsh.exe" set "PWSH=%LOCALAPPDATA%\Microsoft\WindowsApps\pwsh.exe"

if not defined PWSH (
    echo [ERROR] PowerShell 7 ^(pwsh^) not found.
    echo         Install it first:  winget install Microsoft.PowerShell
    pause
    exit /b 1
)

echo Using %PWSH%
"%PWSH%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\build.ps1" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] Build failed with exit code %RC%.
    pause
    exit /b %RC%
)

echo.
echo [OK] Installer is in: %~dp0packaging\Output\
if exist "%~dp0packaging\Output" (
    for %%F in ("%~dp0packaging\Output\pdf-ocrer-setup-*.exe") do echo      %%~fX
)
pause
exit /b 0
