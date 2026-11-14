@echo off
rem ============================================================
rem  pdf-ocrer GUI launcher - double-click to start the app.
rem  1) Uses .venv\Scripts\pythonw.exe when a project venv exists
rem  2) Falls back to pythonw.exe on PATH (no console window)
rem  3) Last resort: python.exe (console stays open for errors)
rem ============================================================
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m pdf_ocrer
    exit /b 0
)

where pythonw.exe >nul 2>nul
if %errorlevel%==0 (
    set "PYTHONPATH=%~dp0src"
    start "" pythonw.exe -m pdf_ocrer
    exit /b 0
)

echo pythonw.exe not found in PATH - starting console python instead...
set "PYTHONPATH=%~dp0src"
python -m pdf_ocrer
