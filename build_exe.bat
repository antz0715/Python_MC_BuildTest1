@echo off
REM ---------------------------------------------------------------------------
REM Build PixelEditor.exe from pixel_editor.py with PyInstaller.
REM Double-click this file to rebuild. The finished executable is left in
REM dist\PixelEditor.exe and needs no Python on the target machine.
REM ---------------------------------------------------------------------------
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set APP_NAME=PixelEditor
set ENTRY=pixel_editor.py
set EXE=dist\%APP_NAME%.exe

echo ===========================================================
echo  Building %APP_NAME%.exe
echo  Folder: %CD%
echo ===========================================================
echo.

REM --- 1. Locate Python ------------------------------------------------------
set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if not defined PY (
    python --version >nul 2>&1 && set PY=python
)
if not defined PY (
    echo [FAILED] Python was not found on this computer.
    echo          Install Python 3 from https://www.python.org/downloads/
    echo          and make sure "py" or "python" works in a new Command Prompt.
    goto :fail
)
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do set PYVER=%%v
echo [1/5] Using %PY%  (%PYVER%)

if not exist "%ENTRY%" (
    echo [FAILED] %ENTRY% was not found next to this batch file.
    goto :fail
)

REM --- 2. Build dependencies -------------------------------------------------
echo [2/5] Installing/updating build dependencies (pyinstaller, pillow)...
%PY% -m pip install --upgrade --disable-pip-version-check pyinstaller pillow
if errorlevel 1 (
    echo [FAILED] Could not install PyInstaller/Pillow.
    echo          Check your internet connection or run this file again.
    goto :fail
)

REM --- 3. Clean previous output ---------------------------------------------
echo [3/5] Removing previous build output...
if exist build rmdir /s /q build
if exist "%APP_NAME%.spec" del /q "%APP_NAME%.spec"
if exist "%EXE%" del /q "%EXE%"

REM --- 4. Build --------------------------------------------------------------
REM To use a custom icon later, drop icon.ico next to this file; it is picked
REM up automatically. (Equivalent to adding --icon=icon.ico by hand.)
set ICON_OPT=
if exist "icon.ico" (
    set ICON_OPT=--icon=icon.ico
    echo [4/5] Building with icon.ico...
) else (
    echo [4/5] Building with the default PyInstaller icon ^(no icon.ico found^)...
)

%PY% -m PyInstaller --noconfirm --clean --onefile --windowed --name %APP_NAME% %ICON_OPT% "%ENTRY%"
if errorlevel 1 (
    echo [FAILED] PyInstaller reported an error. See the output above.
    goto :fail
)

REM --- 5. Verify -------------------------------------------------------------
if not exist "%EXE%" (
    echo [FAILED] PyInstaller finished but %EXE% is missing.
    goto :fail
)
for %%F in ("%EXE%") do set SIZE=%%~zF
set /a SIZE_MB=!SIZE! / 1048576

echo.
echo ===========================================================
echo  [SUCCESS] Build complete.
echo  Executable : %CD%\%EXE%
echo  Size       : !SIZE! bytes (approx. !SIZE_MB! MB)
echo  Launch it by double-clicking dist\%APP_NAME%.exe
echo ===========================================================
echo.
pause
exit /b 0

:fail
echo.
echo ===========================================================
echo  [FAILED] %APP_NAME%.exe was NOT built.
echo ===========================================================
echo.
pause
exit /b 1
