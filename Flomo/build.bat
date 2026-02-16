@echo off
echo Building Flomo...

if not exist build (
    echo Build directory not found. Running configure first...
    call configure.bat
)

cd build
cmake --build . --config Debug

echo.
echo Build complete!
echo Executable: build\Debug\flomo.exe
echo.
pause
