# Flomo dependency setup script (PowerShell wrapper)
# Delegates to setup_dependencies.py for the actual work

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Flomo Dependency Setup" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python using:" -ForegroundColor Yellow
    Write-Host "  winget install Python.Python.3.11"
    Write-Host "  or download from: https://www.python.org/downloads/"
    exit 1
}

Write-Host "[OK] Python found" -ForegroundColor Green
Write-Host ""
Write-Host "Delegating to Python script..." -ForegroundColor Cyan
Write-Host ""

# Run the Python script
Push-Location $scriptDir
python setup_dependencies.py
$exitCode = $LASTEXITCODE
Pop-Location

exit $exitCode
