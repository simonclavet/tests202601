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

echo Configuration succeeded!
echo To test: start build\Flomo.sln
echo Then try building both Debug and Release configurations

pause
