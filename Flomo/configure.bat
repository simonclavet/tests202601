@echo off
echo Flomo Configuration
echo ===================
echo.

REM Check if key dependencies exist, run setup if missing
if not exist "thirdparty\raylib" goto :setup_deps
if not exist "thirdparty\libtorch\debug" goto :setup_deps
if not exist "thirdparty\libtorch\release" goto :setup_deps
goto :run_cmake

:setup_deps
echo Dependencies missing, running setup script...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0setup_dependencies.ps1"
if errorlevel 1 (
    echo.
    echo Dependency setup failed!
    pause
    exit /b 1
)
echo.

:run_cmake
echo Running cmake to create the solution...
echo.

if not exist build mkdir build
cd build

cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES="Debug;RelWithDebInfo" ..

cd ..

echo.
echo Configuration succeeded!
echo.
echo =====================
echo Build Options:
echo =====================
echo.
echo 1. Build from command line (RelWithDebInfo - recommended):
echo    cmake --build build --config RelWithDebInfo
echo.
echo 2. Build from command line (Debug):
echo    cmake --build build --config Debug
echo.
echo 3. Or open Visual Studio solution:
echo    start build\Flomo.sln
echo.
echo =====================
echo Run the program:
echo =====================
echo.
echo After building, run directly:
echo    build\src\RelWithDebInfo\flomo.exe
echo or:
echo    build\src\Debug\flomo.exe
echo.

pause
