@echo off
REM ============================================
REM PyCAM3D - Windows Executable Builder
REM ============================================

echo.
echo ========================================
echo   PyCAM3D - Building Windows Executable
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -e . --quiet
pip install pyinstaller --quiet

REM Build executable
echo.
echo Building executable...
pyinstaller --onefile ^
    --name pycam3d ^
    --icon=NONE ^
    --hidden-import=trimesh ^
    --hidden-import=scipy ^
    --hidden-import=scipy.spatial ^
    --hidden-import=scipy.spatial.transform ^
    --hidden-import=numpy ^
    --hidden-import=click ^
    --hidden-import=rich ^
    --hidden-import=pycam3d ^
    --hidden-import=pycam3d.cli ^
    --hidden-import=pycam3d.chunk ^
    --hidden-import=pycam3d.patterns ^
    --hidden-import=pycam3d.patterns.rotary ^
    --hidden-import=pycam3d.collision ^
    --hidden-import=pycam3d.collision.sampler_4axis ^
    --hidden-import=pycam3d.gcode_multiaxis ^
    --hidden-import=pycam3d.pipeline_4axis ^
    --exclude-module=tkinter ^
    --exclude-module=matplotlib ^
    --exclude-module=IPython ^
    --exclude-module=jupyter ^
    --exclude-module=notebook ^
    --exclude-module=pytest ^
    src\pycam3d\cli.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   BUILD SUCCESS!
echo ========================================
echo.
echo Executable created: dist\pycam3d.exe
echo.
echo Usage:
echo   dist\pycam3d.exe --help
echo   dist\pycam3d.exe 4axis piece.stl -o output.nc
echo.

REM Copy to root
copy dist\pycam3d.exe . >nul 2>&1

echo Copied to: pycam3d.exe
echo.
pause
