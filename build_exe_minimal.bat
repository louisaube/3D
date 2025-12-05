@echo off
REM ============================================
REM PyCAM3D - Minimal Windows Executable Builder
REM (Excludes heavy dependencies for faster build)
REM ============================================

echo.
echo ========================================
echo   PyCAM3D - Building Minimal Executable
echo ========================================
echo.

REM Install core dependencies only
echo Installing core dependencies...
pip install numpy trimesh scipy click rich pyinstaller --quiet

REM Create a minimal entry point
echo Creating entry point...
echo import sys > _entry.py
echo sys.path.insert(0, 'src') >> _entry.py
echo from pycam3d.cli import main >> _entry.py
echo if __name__ == '__main__': main() >> _entry.py

REM Build executable
echo.
echo Building executable (this may take a few minutes)...
pyinstaller --onefile ^
    --name pycam3d ^
    --paths=src ^
    --hidden-import=pycam3d ^
    --hidden-import=pycam3d.cli ^
    --hidden-import=pycam3d.chunk ^
    --hidden-import=pycam3d.patterns.rotary ^
    --hidden-import=pycam3d.collision.sampler_4axis ^
    --hidden-import=pycam3d.gcode_multiaxis ^
    --hidden-import=pycam3d.pipeline_4axis ^
    --hidden-import=pycam3d.tool ^
    --hidden-import=pycam3d.toolpath ^
    --hidden-import=trimesh ^
    --hidden-import=scipy.spatial.transform ^
    --collect-submodules=pycam3d ^
    --exclude-module=open3d ^
    --exclude-module=opencamlib ^
    --exclude-module=tkinter ^
    --exclude-module=matplotlib ^
    --exclude-module=IPython ^
    --exclude-module=jupyter ^
    --exclude-module=pandas ^
    --exclude-module=PIL ^
    _entry.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    del _entry.py 2>nul
    pause
    exit /b 1
)

REM Cleanup
del _entry.py 2>nul
move dist\_entry.exe dist\pycam3d.exe >nul 2>&1

echo.
echo ========================================
echo   BUILD SUCCESS!
echo ========================================
echo.
echo Executable: dist\pycam3d.exe
echo.
copy dist\pycam3d.exe . >nul 2>&1
echo.
pause
