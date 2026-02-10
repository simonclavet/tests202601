#!/usr/bin/env python3
"""
Flomo dependency setup script
Downloads all thirdparty dependencies if they don't exist
"""

import os
import sys
import subprocess
import shutil
import urllib.request
import zipfile
import re

# Configuration
LIBTORCH_VERSION = "2.10.0"

# Color output helpers
class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    RESET = '\033[0m'

    @staticmethod
    def enabled():
        """Check if color output is supported."""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

def print_color(text, color):
    """Print colored text if terminal supports it."""
    if Color.enabled():
        print(f"{color}{text}{Color.RESET}")
    else:
        print(text)

def run_command(cmd):
    """Run command and return output and return code."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout.strip(), result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return "", 1

def check_command_exists(cmd):
    """Check if a command exists in PATH."""
    output, ret = run_command(f"{'where' if os.name == 'nt' else 'which'} {cmd}")
    return ret == 0

def parse_version(version_str):
    """Parse version string like '3.21.0' into tuple."""
    match = re.search(r'(\d+)\.(\d+)(?:\.(\d+))?', version_str)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3)) if match.group(3) else 0
        return (major, minor, patch)
    return None

def clone_if_missing(folder, url, thirdparty_dir):
    """Clone git repo if it doesn't exist."""
    path = os.path.join(thirdparty_dir, folder)
    if os.path.exists(path):
        print_color(f"[OK] {folder} already exists", Color.GREEN)
        return True

    print_color(f"[DOWNLOADING] {folder} from {url}", Color.YELLOW)
    cmd = f'git clone --depth 1 "{url}" "{path}"'
    _, ret = run_command(cmd)

    if ret != 0:
        print_color(f"[ERROR] Failed to clone {folder}", Color.RED)
        return False

    print_color(f"[OK] {folder} cloned", Color.GREEN)
    return True

def download_file(url, dest_path):
    """Download file with progress."""
    print(f"  Downloading: {url}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, report_progress)
    print()  # New line after progress

def get_cuda_version():
    """Get installed CUDA version."""
    output, ret = run_command("nvcc --version")
    if ret != 0:
        return None

    # Parse version from: "release 13.1, V13.1.115"
    match = re.search(r'release (\d+\.\d+)', output)
    if match:
        version = match.group(1)
        major, minor = map(int, version.split('.'))
        return (major, minor)
    return None

def get_gpu_compute_capability():
    """Get GPU compute capability."""
    output, ret = run_command(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader"
    )
    if ret != 0:
        return None

    try:
        # Parse "12.0" -> (12, 0)
        cap = output.strip().split('\n')[0]
        major, minor = map(int, cap.split('.'))
        return (major, minor)
    except:
        return None

def select_libtorch_cuda_version(cuda_version, compute_cap):
    """
    Select appropriate LibTorch CUDA version.

    Available LibTorch 2.10.0 options: cu126, cu128, cu130
    Blackwell (SM 12.0) requires CUDA 12.6+
    """
    if not cuda_version:
        print("No CUDA installation found, using CPU version")
        return "cpu"

    if not compute_cap:
        print("Cannot detect GPU, using CPU version")
        return "cpu"

    major_cuda, minor_cuda = cuda_version
    major_cap, minor_cap = compute_cap

    print(f"Detected: CUDA {major_cuda}.{minor_cuda}, "
          f"GPU compute capability {major_cap}.{minor_cap}")

    # Blackwell (SM 12.0) requires CUDA 12.6+
    if major_cap >= 12:
        # Check if CUDA version is new enough
        # (major >= 13) OR (major == 12 AND minor >= 6)
        if major_cuda >= 13 or (major_cuda == 12 and minor_cuda >= 6):
            # Match CUDA version to LibTorch version
            if major_cuda >= 13:
                return "cu130"
            elif minor_cuda >= 8:
                return "cu128"
            else:
                return "cu126"
        else:
            print(f"WARNING: Blackwell GPU detected but CUDA version "
                  f"{major_cuda}.{minor_cuda} is too old (need 12.6+)")
            print("Please upgrade CUDA to 12.6 or newer")
            return "cpu"

    # For older GPUs, match CUDA version
    if major_cuda >= 13:
        return "cu130"
    elif major_cuda == 12:
        if minor_cuda >= 8:
            return "cu128"
        else:
            return "cu126"
    else:
        print(f"WARNING: CUDA {major_cuda}.{minor_cuda} is too old")
        return "cpu"

def get_libtorch_url(cuda_version, debug=False):
    """Get LibTorch download URL for Windows."""
    base_url = "https://download.pytorch.org/libtorch"
    debug_suffix = "-debug" if debug else ""

    if cuda_version == "cpu":
        return (
            f"{base_url}/cpu/libtorch-win-shared-with-deps{debug_suffix}-"
            f"{LIBTORCH_VERSION}%2Bcpu.zip"
        )
    else:
        return (
            f"{base_url}/{cuda_version}/libtorch-win-shared-with-deps"
            f"{debug_suffix}-{LIBTORCH_VERSION}%2B{cuda_version}.zip"
        )

def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"  Extracting to: {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def setup_libtorch_build(target_dir, cuda_version, debug=False):
    """Download and setup one LibTorch build (debug or release)."""

    build_name = "DEBUG" if debug else "RELEASE"

    # Try requested version first
    versions_to_try = [cuda_version]

    # Add fallbacks if CUDA version fails
    if cuda_version == "cu130":
        versions_to_try.extend(["cu128", "cu126"])
    elif cuda_version == "cu128":
        versions_to_try.append("cu126")

    last_error = None

    for version in versions_to_try:
        url = get_libtorch_url(version, debug)
        temp_zip = os.path.join(target_dir, "libtorch_temp.zip")

        try:
            if version != cuda_version:
                print(f"\n  Trying fallback version: {version}")

            download_file(url, temp_zip)

            # Extract
            extract_zip(temp_zip, target_dir)
            os.remove(temp_zip)

            # The zip extracts to libtorch/
            extracted_path = os.path.join(target_dir, "libtorch")

            if not os.path.exists(extracted_path):
                print("  ERROR: Extracted libtorch directory not found")
                continue

            if version != cuda_version:
                print(f"  Successfully downloaded {version} as fallback")

            return extracted_path

        except Exception as e:
            last_error = e
            print(f"  Failed to download {build_name} {version}: {e}")
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            continue

    print(f"\n  ERROR: All {build_name} download attempts failed. "
          f"Last error: {last_error}")
    return False

def setup_libtorch(thirdparty_dir):
    """Setup LibTorch with auto-detection of GPU and CUDA version."""

    # Detect system
    cuda_version = get_cuda_version()
    compute_cap = get_gpu_compute_capability()

    # Select appropriate LibTorch version
    libtorch_cuda = select_libtorch_cuda_version(cuda_version, compute_cap)

    print(f"\nSelected LibTorch variant: {libtorch_cuda}")

    # Check what's already installed
    libtorch_dir = os.path.join(thirdparty_dir, "libtorch")
    release_dir = os.path.join(libtorch_dir, "release")
    debug_dir = os.path.join(libtorch_dir, "debug")

    has_release = os.path.exists(release_dir)
    has_debug = os.path.exists(debug_dir)

    if has_release and has_debug:
        print_color("[OK] libtorch (release and debug) already exists",
                   Color.GREEN)
        print_color("  To update, delete 'thirdparty/libtorch' and run again",
                   Color.GRAY)
        return True

    need_release = not has_release
    need_debug = not has_debug

    # Download RELEASE if needed
    if need_release:
        print("\nDownloading LibTorch RELEASE...")
        extracted_release = setup_libtorch_build(
            thirdparty_dir, libtorch_cuda, debug=False
        )

        if not extracted_release:
            print("\nFailed to setup LibTorch RELEASE")
            return False

        # Move to final location immediately (so progress is saved if interrupted)
        os.makedirs(os.path.dirname(release_dir), exist_ok=True)
        if os.path.exists(release_dir):
            shutil.rmtree(release_dir)
        shutil.move(extracted_release, release_dir)
        print_color("[OK] LibTorch RELEASE installed", Color.GREEN)
    elif need_debug:
        print_color("[OK] LibTorch RELEASE already exists", Color.GREEN)

    # Download DEBUG if needed
    if need_debug:
        print("\nDownloading LibTorch DEBUG...")
        extracted_debug = setup_libtorch_build(
            thirdparty_dir, libtorch_cuda, debug=True
        )

        if extracted_debug:
            # Move to final location immediately
            os.makedirs(os.path.dirname(debug_dir), exist_ok=True)
            if os.path.exists(debug_dir):
                shutil.rmtree(debug_dir)
            shutil.move(extracted_debug, debug_dir)
            print_color("[OK] LibTorch DEBUG installed", Color.GREEN)
        else:
            print("\nWARNING: Failed to setup LibTorch DEBUG")
            print("Debug builds will use release libraries (may cause issues)")
    elif need_release:
        print_color("[OK] LibTorch DEBUG already exists", Color.GREEN)

    # Verify and print summary
    print()
    final_has_release = os.path.exists(release_dir)
    final_has_debug = os.path.exists(debug_dir)

    if not final_has_release:
        print_color("[ERROR] LibTorch setup incomplete", Color.RED)
        return False

    print_color(f"[OK] LibTorch ({libtorch_cuda}) ready!", Color.GREEN)
    print(f"  Release: {release_dir}")
    if final_has_debug:
        print(f"  Debug: {debug_dir}")
    else:
        print("  Debug: Not available (using release for debug builds)")

    # Show version info
    build_version_path = os.path.join(release_dir, "build-version")
    if os.path.exists(build_version_path):
        with open(build_version_path, 'r') as f:
            version = f.read().strip()
            print(f"  Version: {version}")

    return True

def main():
    """Main setup function."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    thirdparty_dir = os.path.join(script_dir, "thirdparty")

    # Required versions
    required_cmake_version = (3, 21, 0)
    required_cuda_version = (13, 1, 0)

    print_color("Flomo Dependency Setup", Color.CYAN)
    print_color("======================", Color.CYAN)
    print()

    # Check prerequisites
    print_color("Checking prerequisites...", Color.CYAN)

    # Check CMake
    if not check_command_exists("cmake"):
        print_color("[MISSING] CMake not found!", Color.RED)
        print()
        print_color("Install CMake 3.21+ using one of these methods:", Color.YELLOW)
        print("  winget install Kitware.CMake")
        print("  or download from: https://cmake.org/download/")
        print()
        print_color("Make sure CMake is added to your PATH.", Color.YELLOW)
        return 1

    output, _ = run_command("cmake --version")
    version = parse_version(output)
    if version and version < required_cmake_version:
        print_color(f"[ERROR] CMake {version[0]}.{version[1]} found, but "
                   f"{required_cmake_version[0]}.{required_cmake_version[1]}+ "
                   f"required", Color.RED)
        print("  winget upgrade Kitware.CMake")
        return 1
    if version:
        print_color(f"[OK] CMake {version[0]}.{version[1]}.{version[2]}",
                   Color.GREEN)
    else:
        print_color("[WARNING] Could not parse CMake version, continuing...",
                   Color.YELLOW)

    # Check CUDA (nvcc)
    if not check_command_exists("nvcc"):
        print_color("[MISSING] CUDA (nvcc) not found!", Color.RED)
        print()
        print_color("This project requires CUDA 13.1+ with SM 86 support "
                   "(RTX 30 series+).", Color.YELLOW)
        print()
        print_color("Download CUDA Toolkit directly (no login required):",
                   Color.YELLOW)
        print_color("  https://developer.download.nvidia.com/compute/cuda/"
                   "13.1.1/local_installers/cuda_13.1.1_windows.exe",
                   Color.CYAN)
        print()
        print_color("After installation, make sure nvcc is in your PATH.",
                   Color.YELLOW)
        print_color("Default location: C:\\Program Files\\NVIDIA GPU Computing "
                   "Toolkit\\CUDA\\v13.1\\bin", Color.GRAY)
        return 1

    output, _ = run_command("nvcc --version")
    version = parse_version(output)
    if version and version < required_cuda_version:
        print_color(f"[ERROR] CUDA {version[0]}.{version[1]} found, but "
                   f"{required_cuda_version[0]}.{required_cuda_version[1]}+ "
                   f"required", Color.RED)
        print("  Download: https://developer.download.nvidia.com/compute/"
              "cuda/13.1.1/local_installers/cuda_13.1.1_windows.exe")
        return 1
    if version:
        print_color(f"[OK] CUDA {version[0]}.{version[1]}", Color.GREEN)
    else:
        print_color("[WARNING] Could not parse CUDA version, continuing...",
                   Color.YELLOW)

    # Check Git
    if not check_command_exists("git"):
        print_color("[MISSING] Git not found!", Color.RED)
        print()
        print_color("Install Git using:", Color.YELLOW)
        print("  winget install Git.Git")
        print("  or download from: https://git-scm.com/download/win")
        return 1
    print_color("[OK] Git found", Color.GREEN)

    print()
    print(f"Thirdparty dir: {thirdparty_dir}")
    print()

    # Create thirdparty folder if needed
    if not os.path.exists(thirdparty_dir):
        os.makedirs(thirdparty_dir)
        print("Created thirdparty folder")

    # Git repos
    print()
    print_color("Checking git dependencies...", Color.CYAN)

    repos = [
        ("raylib", "https://github.com/raysan5/raylib.git"),
        ("raygui", "https://github.com/raysan5/raygui.git"),
        ("imgui", "https://github.com/ocornut/imgui.git"),
        ("rlImGui", "https://github.com/raylib-extras/rlImGui.git"),
    ]

    for folder, url in repos:
        if not clone_if_missing(folder, url, thirdparty_dir):
            return 1

    # tiny-cuda-nn - fast neural networks for motion matching
    print()
    print_color("Checking tiny-cuda-nn...", Color.CYAN)
    tiny_cuda_nn_dir = os.path.join(thirdparty_dir, "tiny-cuda-nn")

    if os.path.exists(tiny_cuda_nn_dir):
        print_color("[OK] tiny-cuda-nn already exists", Color.GREEN)
    else:
        print_color("[DOWNLOADING] tiny-cuda-nn (with submodules)",
                   Color.YELLOW)
        cmd = f'git clone --recursive --depth 1 ' \
              f'https://github.com/NVlabs/tiny-cuda-nn.git "{tiny_cuda_nn_dir}"'
        _, ret = run_command(cmd)

        if ret != 0:
            print_color("[ERROR] Failed to clone tiny-cuda-nn", Color.RED)
            return 1
        print_color("[OK] tiny-cuda-nn cloned", Color.GREEN)

    # Verify tiny-cuda-nn submodules are initialized
    if os.path.exists(tiny_cuda_nn_dir):
        result = subprocess.run(
            "git submodule status",
            shell=True,
            capture_output=True,
            text=True,
            cwd=tiny_cuda_nn_dir
        )
        output = result.stdout.strip()

        if output.startswith("-") or "No submodule" in output:
            print_color("[UPDATING] Initializing tiny-cuda-nn submodules...",
                       Color.YELLOW)
            result = subprocess.run(
                "git submodule update --init --recursive --depth 1",
                shell=True,
                cwd=tiny_cuda_nn_dir
            )
            if result.returncode != 0:
                print_color("[WARNING] Some submodules may have failed to "
                           "initialize", Color.YELLOW)
            else:
                print_color("[OK] Submodules initialized", Color.GREEN)

    # ufbx - just two files
    print()
    print_color("Checking ufbx...", Color.CYAN)
    ufbx_dir = os.path.join(thirdparty_dir, "ufbx")

    if os.path.exists(ufbx_dir):
        print_color("[OK] ufbx already exists", Color.GREEN)
    else:
        print_color("[DOWNLOADING] ufbx source files", Color.YELLOW)
        os.makedirs(ufbx_dir)

        base_url = "https://raw.githubusercontent.com/ufbx/ufbx/master"

        try:
            download_file(f"{base_url}/ufbx.h",
                         os.path.join(ufbx_dir, "ufbx.h"))
            download_file(f"{base_url}/ufbx.c",
                         os.path.join(ufbx_dir, "ufbx.c"))
            print_color("[OK] ufbx downloaded", Color.GREEN)
        except Exception as e:
            print_color(f"[ERROR] Failed to download ufbx: {e}", Color.RED)
            return 1

    # libtorch - auto-detect GPU and download appropriate version
    print()
    print_color("Checking libtorch...", Color.CYAN)

    success = setup_libtorch(thirdparty_dir)
    if not success:
        print_color("[ERROR] LibTorch setup failed", Color.RED)
        return 1

    print()
    print_color("All dependencies ready!", Color.GREEN)
    print_color("Run configure.bat to generate the Visual Studio solution.",
               Color.CYAN)

    return 0

if __name__ == "__main__":
    sys.exit(main())
