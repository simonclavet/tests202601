# Flomo dependency setup script
# Downloads all thirdparty dependencies if they don't exist

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$thirdpartyDir = Join-Path $scriptDir "thirdparty"

# libtorch version config
$torchVersion = "2.5.1"
$cudaVersion = "cu121"

# required versions
$requiredCmakeVersion = [Version]"3.21"
$requiredCudaVersion = [Version]"13.1"

Write-Host "Flomo Dependency Setup" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host ""

# check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Cyan

# check cmake
$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakeCmd) {
    Write-Host "[MISSING] CMake not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install CMake 3.21+ using one of these methods:" -ForegroundColor Yellow
    Write-Host "  winget install Kitware.CMake"
    Write-Host "  or download from: https://cmake.org/download/"
    Write-Host ""
    Write-Host "Make sure CMake is added to your PATH." -ForegroundColor Yellow
    exit 1
}

$cmakeVersionOutput = cmake --version | Select-Object -First 1
if ($cmakeVersionOutput -match "cmake version (\d+\.\d+(\.\d+)?)") {
    $cmakeVersion = [Version]$Matches[1]
    if ($cmakeVersion -lt $requiredCmakeVersion) {
        Write-Host "[ERROR] CMake $cmakeVersion found, but $requiredCmakeVersion+ required" -ForegroundColor Red
        Write-Host "  winget upgrade Kitware.CMake"
        exit 1
    }
    Write-Host "[OK] CMake $cmakeVersion" -ForegroundColor Green
}
else {
    Write-Host "[WARNING] Could not parse CMake version, continuing anyway..." -ForegroundColor Yellow
}

# check cuda (nvcc)
$nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvccCmd) {
    Write-Host "[MISSING] CUDA (nvcc) not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "This project requires CUDA 13.1+ with SM 86 support (RTX 30 series+)." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Download CUDA Toolkit directly (no login required):" -ForegroundColor Yellow
    Write-Host "  https://developer.download.nvidia.com/compute/cuda/13.1.1/local_installers/cuda_13.1.1_windows.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "After installation, make sure nvcc is in your PATH." -ForegroundColor Yellow
    Write-Host "Default location: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin" -ForegroundColor Gray
    exit 1
}

$nvccVersionOutput = nvcc --version | Select-Object -Last 1
if ($nvccVersionOutput -match "release (\d+\.\d+)") {
    $cudaVersionInstalled = [Version]$Matches[1]
    if ($cudaVersionInstalled -lt $requiredCudaVersion) {
        Write-Host "[ERROR] CUDA $cudaVersionInstalled found, but $requiredCudaVersion+ required" -ForegroundColor Red
        Write-Host "  Download: https://developer.download.nvidia.com/compute/cuda/13.1.1/local_installers/cuda_13.1.1_windows.exe"
        exit 1
    }
    Write-Host "[OK] CUDA $cudaVersionInstalled" -ForegroundColor Green
}
else {
    Write-Host "[WARNING] Could not parse CUDA version, continuing anyway..." -ForegroundColor Yellow
}

# check git (needed for cloning repos)
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "[MISSING] Git not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Git using:" -ForegroundColor Yellow
    Write-Host "  winget install Git.Git"
    Write-Host "  or download from: https://git-scm.com/download/win"
    exit 1
}
Write-Host "[OK] Git found" -ForegroundColor Green

Write-Host ""
Write-Host "Thirdparty dir: $thirdpartyDir"
Write-Host ""

# Create thirdparty folder if needed
if (-not (Test-Path $thirdpartyDir)) {
    New-Item -ItemType Directory -Path $thirdpartyDir | Out-Null
    Write-Host "Created thirdparty folder"
}

Set-Location $thirdpartyDir

# Helper function to clone a git repo if missing
function Clone-IfMissing {
    param (
        [string]$folder,
        [string]$url
    )

    $path = Join-Path $thirdpartyDir $folder
    if (Test-Path $path) {
        Write-Host "[OK] $folder already exists" -ForegroundColor Green
        return
    }

    Write-Host "[DOWNLOADING] $folder from $url" -ForegroundColor Yellow
    git clone --depth 1 $url $folder
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone $folder"
    }
    Write-Host "[OK] $folder cloned" -ForegroundColor Green
}

# Helper function to download and extract a zip
function Download-AndExtract {
    param (
        [string]$url,
        [string]$destFolder,
        [string]$description
    )

    $zipFile = Join-Path $env:TEMP "download_temp.zip"

    Write-Host "[DOWNLOADING] $description" -ForegroundColor Yellow
    Write-Host "  URL: $url"

    # Download with progress
    $ProgressPreference = 'SilentlyContinue'  # Speeds up Invoke-WebRequest significantly
    Invoke-WebRequest -Uri $url -OutFile $zipFile -UseBasicParsing

    Write-Host "[EXTRACTING] $description to $destFolder" -ForegroundColor Yellow

    # Extract to temp folder first (zip contains 'libtorch' folder)
    $tempExtract = Join-Path $env:TEMP "extract_temp"
    if (Test-Path $tempExtract) {
        Remove-Item -Recurse -Force $tempExtract
    }

    Expand-Archive -Path $zipFile -DestinationPath $tempExtract -Force

    # Move the contents (libtorch folder) to destination
    $extractedFolder = Join-Path $tempExtract "libtorch"
    if (-not (Test-Path $extractedFolder)) {
        throw "Expected 'libtorch' folder in zip, but not found"
    }

    # Create dest and move contents
    if (-not (Test-Path $destFolder)) {
        New-Item -ItemType Directory -Path $destFolder | Out-Null
    }

    Get-ChildItem -Path $extractedFolder | Move-Item -Destination $destFolder -Force

    # Cleanup
    Remove-Item -Recurse -Force $tempExtract
    Remove-Item -Force $zipFile

    Write-Host "[OK] $description installed" -ForegroundColor Green
}

# Git repos
Write-Host ""
Write-Host "Checking git dependencies..." -ForegroundColor Cyan
Clone-IfMissing "raylib" "https://github.com/raysan5/raylib.git"
Clone-IfMissing "raygui" "https://github.com/raysan5/raygui.git"
Clone-IfMissing "imgui" "https://github.com/ocornut/imgui.git"
Clone-IfMissing "rlImGui" "https://github.com/raylib-extras/rlImGui.git"


# tiny-cuda-nn - fast neural networks for motion matching
Write-Host ""
Write-Host "Checking tiny-cuda-nn..." -ForegroundColor Cyan
$tinyCudaNNDir = Join-Path $thirdpartyDir "tiny-cuda-nn"
if (Test-Path $tinyCudaNNDir) {
    Write-Host "[OK] tiny-cuda-nn already exists" -ForegroundColor Green
} else {
    Write-Host "[DOWNLOADING] tiny-cuda-nn (with submodules)" -ForegroundColor Yellow
    git clone --recursive --depth 1 https://github.com/NVlabs/tiny-cuda-nn.git $tinyCudaNNDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone tiny-cuda-nn"
    }
    Write-Host "[OK] tiny-cuda-nn cloned" -ForegroundColor Green
}

# Verify tiny-cuda-nn submodules are initialized
if (Test-Path $tinyCudaNNDir) {
    Push-Location $tinyCudaNNDir
    
    # Check if submodules need initialization
    $submoduleStatus = git submodule status 2>&1
    if ($submoduleStatus -match "^-" -or $submoduleStatus -match "No submodule") {
        Write-Host "[UPDATING] Initializing tiny-cuda-nn submodules..." -ForegroundColor Yellow
        git submodule update --init --recursive --depth 1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARNING] Some submodules may have failed to initialize" -ForegroundColor Yellow
        } else {
            Write-Host "[OK] Submodules initialized" -ForegroundColor Green
        }
    }
    
    Pop-Location
}




# ufbx - just two files
Write-Host ""
Write-Host "Checking ufbx..." -ForegroundColor Cyan
$ufbxDir = Join-Path $thirdpartyDir "ufbx"
if (Test-Path $ufbxDir) {
    Write-Host "[OK] ufbx already exists" -ForegroundColor Green
} else {
    Write-Host "[DOWNLOADING] ufbx source files" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $ufbxDir | Out-Null

    $ProgressPreference = 'SilentlyContinue'
    $baseUrl = "https://raw.githubusercontent.com/ufbx/ufbx/master"
    Invoke-WebRequest -Uri "$baseUrl/ufbx.h" -OutFile (Join-Path $ufbxDir "ufbx.h") -UseBasicParsing
    Invoke-WebRequest -Uri "$baseUrl/ufbx.c" -OutFile (Join-Path $ufbxDir "ufbx.c") -UseBasicParsing

    Write-Host "[OK] ufbx downloaded" -ForegroundColor Green
}

# libtorch - need both debug and release
Write-Host ""
Write-Host "Checking libtorch..." -ForegroundColor Cyan
$libtorchDir = Join-Path $thirdpartyDir "libtorch"
$debugDir = Join-Path $libtorchDir "debug"
$releaseDir = Join-Path $libtorchDir "release"

# Create libtorch folder if needed
if (-not (Test-Path $libtorchDir)) {
    New-Item -ItemType Directory -Path $libtorchDir | Out-Null
}

# Debug build
if (Test-Path $debugDir) {
    Write-Host "[OK] libtorch/debug already exists" -ForegroundColor Green
} else {
    $debugUrl = "https://download.pytorch.org/libtorch/$cudaVersion/libtorch-win-shared-with-deps-debug-$torchVersion%2B$cudaVersion.zip"
    Download-AndExtract $debugUrl $debugDir "libtorch DEBUG ($torchVersion+$cudaVersion)"
}

# Release build
if (Test-Path $releaseDir) {
    Write-Host "[OK] libtorch/release already exists" -ForegroundColor Green
} else {
    $releaseUrl = "https://download.pytorch.org/libtorch/$cudaVersion/libtorch-win-shared-with-deps-$torchVersion%2B$cudaVersion.zip"
    Download-AndExtract $releaseUrl $releaseDir "libtorch RELEASE ($torchVersion+$cudaVersion)"
}

Write-Host ""
Write-Host "All dependencies ready!" -ForegroundColor Green
Write-Host "Run configure.bat to generate the Visual Studio solution." -ForegroundColor Cyan
