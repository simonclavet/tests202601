@echo off
echo Building Flomo...

if not exist build (
    echo Build directory not found. Running configure first...
    call configure.bat
)

cd build
cmake --build . --config Release

echo.
echo Build complete!
echo Executable: build\Release\flomo.exe
echo.
pause
