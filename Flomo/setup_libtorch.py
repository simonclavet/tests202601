#!/usr/bin/env python3
"""
Download and setup LibTorch with appropriate CUDA version for the system.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
import re

def run_command(cmd):
    """Run command and return output."""
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

def get_cuda_version():
    """Get installed CUDA version."""
    output, ret = run_command("nvcc --version")
    if ret != 0:
        print("WARNING: nvcc not found, CUDA may not be installed")
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
        print("WARNING: nvidia-smi failed, cannot detect GPU")
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

def get_libtorch_url(cuda_version):
    """Get LibTorch download URL for Windows."""
    base_url = "https://download.pytorch.org/libtorch"
    version = "2.10.0"  # Latest stable

    if cuda_version == "cpu":
        return (
            f"{base_url}/cpu/libtorch-win-shared-with-deps-{version}"
            f"%2Bcpu.zip"
        )
    else:
        return (
            f"{base_url}/{cuda_version}/libtorch-win-shared-with-deps-"
            f"{version}%2B{cuda_version}.zip"
        )

def download_file(url, dest_path):
    """Download file with progress."""
    print(f"Downloading: {url}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\rProgress: {percent:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, report_progress)
    print()  # New line after progress

def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"Extracting to: {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def setup_libtorch(target_dir, cuda_version):
    """Download and setup LibTorch with fallback."""

    # Try requested version first
    versions_to_try = [cuda_version]

    # Add fallbacks if CUDA version fails
    if cuda_version == "cu130":
        versions_to_try.extend(["cu128", "cu126"])
    elif cuda_version == "cu128":
        versions_to_try.append("cu126")

    last_error = None

    for version in versions_to_try:
        url = get_libtorch_url(version)
        temp_zip = os.path.join(target_dir, "libtorch_temp.zip")

        try:
            if version != cuda_version:
                print(f"\nTrying fallback version: {version}")

            download_file(url, temp_zip)

            # Extract
            extract_zip(temp_zip, target_dir)
            os.remove(temp_zip)

            # The zip extracts to libtorch/
            extracted_path = os.path.join(target_dir, "libtorch")

            if not os.path.exists(extracted_path):
                print("ERROR: Extracted libtorch directory not found")
                continue

            if version != cuda_version:
                print(f"Successfully downloaded {version} as fallback")

            return extracted_path

        except Exception as e:
            last_error = e
            print(f"Failed to download {version}: {e}")
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            continue

    print(f"\nERROR: All download attempts failed. Last error: {last_error}")
    return False

def main():
    """Main setup function."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    thirdparty_dir = os.path.join(script_dir, "thirdparty")

    # Detect system
    cuda_version = get_cuda_version()
    compute_cap = get_gpu_compute_capability()

    # Select appropriate LibTorch version
    libtorch_cuda = select_libtorch_cuda_version(cuda_version, compute_cap)

    print(f"\nSelected LibTorch variant: {libtorch_cuda}")

    # Ask user confirmation (skip if auto-confirm env var set)
    auto_confirm = os.environ.get("LIBTORCH_AUTO_CONFIRM", "").lower() == "1"
    if not auto_confirm:
        response = input("\nDownload and install LibTorch? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    else:
        print("\n[Auto-confirmed] Proceeding with LibTorch installation...")

    # Backup existing libtorch if it exists
    libtorch_dir = os.path.join(thirdparty_dir, "libtorch")
    if os.path.exists(libtorch_dir):
        backup_dir = libtorch_dir + ".backup"
        print(f"\nBacking up existing LibTorch to: {backup_dir}")
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(libtorch_dir, backup_dir)

    # Download and setup
    print("\nDownloading LibTorch...")
    extracted_path = setup_libtorch(thirdparty_dir, libtorch_cuda)

    if not extracted_path:
        print("\nFailed to setup LibTorch")
        return 1

    # Reorganize into release/debug structure
    # extracted_path is thirdparty/libtorch (from zip)
    # We need: thirdparty/libtorch/release/ and thirdparty/libtorch/debug/

    temp_path = os.path.join(thirdparty_dir, "libtorch_temp")

    # Rename extracted libtorch to temp location
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    shutil.move(extracted_path, temp_path)

    # Create new libtorch directory with release subdirectory
    release_dir = os.path.join(thirdparty_dir, "libtorch", "release")
    os.makedirs(os.path.dirname(release_dir), exist_ok=True)

    # Move temp to release
    shutil.move(temp_path, release_dir)

    # Note about debug build
    debug_dir = os.path.join(thirdparty_dir, "libtorch", "debug")
    print("\nNote: Using release build for debug config")
    print("(Debug LibTorch would require separate download)")

    print(f"\n✓ LibTorch ({libtorch_cuda}) installed successfully!")
    print(f"Location: {release_dir}")

    # Write version info
    build_version_path = os.path.join(release_dir, "build-version")
    if os.path.exists(build_version_path):
        with open(build_version_path, 'r') as f:
            version = f.read().strip()
            print(f"Version: {version}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
