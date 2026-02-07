# Flomo - BVH/FBX Animation Viewer
# Root CMakeLists.txt - Project configuration and subdirectories

cmake_minimum_required(VERSION 3.21)

# MSVC runtime configuration (before project())
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
endif()

# Only Debug and RelWithDebInfo configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo" CACHE STRING "" FORCE)

# =============================================================================
# Windows SDK compatibility fix (MUST be before project())
# =============================================================================
# Force using Windows 10 SDK 22H2 instead of the latest 24H2
# The newer SDK (10.0.26100.0) has compatibility issues
set(CMAKE_SYSTEM_VERSION "10.0.22621.0" CACHE STRING "Windows SDK version" FORCE)

project(Flomo LANGUAGES C CXX CUDA)

# =============================================================================
# Global settings
# =============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU compute capability using nvidia-smi
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_COMPUTE_CAP
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(GPU_COMPUTE_CAP)
    # Convert "8.6" -> "86", "8.9" -> "89", etc.
    string(REPLACE "." "" CUDA_ARCH_NUMBER "${GPU_COMPUTE_CAP}")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_NUMBER}")
    message(STATUS "Detected GPU compute capability: ${GPU_COMPUTE_CAP} -> arch ${CUDA_ARCH_NUMBER}")
else()
    # Fallback if nvidia-smi fails
    set(CMAKE_CUDA_ARCHITECTURES "86")
    message(WARNING "Could not detect GPU, using default architecture: 86")
endif()

# =============================================================================
# Libtorch paths (shared between targets)
# =============================================================================

set(TORCH_DEBUG_PATH "${CMAKE_SOURCE_DIR}/thirdparty/libtorch/debug")
set(TORCH_RELEASE_PATH "${CMAKE_SOURCE_DIR}/thirdparty/libtorch/release")
set(TORCH_INCLUDE_DIR "${TORCH_RELEASE_PATH}/include")
set(TORCH_INCLUDE_DIR_2 "${TORCH_RELEASE_PATH}/include/torch/csrc/api/include")

# =============================================================================
# Optional: tiny-cuda-nn (set to ON to enable)
# =============================================================================

option(USE_TINY_CUDA_NN "Enable tiny-cuda-nn neural network library" OFF)

if(USE_TINY_CUDA_NN)
    message(STATUS "tiny-cuda-nn: ENABLED")

    # Configure tiny-cuda-nn options BEFORE add_subdirectory
    set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(TCNN_ENABLE_RTC OFF CACHE BOOL "" FORCE)  # Disable runtime compilation to avoid conflicts
    # Don't set TCNN_CUDA_ARCHITECTURES - let it inherit from CMAKE_CUDA_ARCHITECTURES

    # Add tiny-cuda-nn (creates the 'tiny-cuda-nn' target)
    add_subdirectory(thirdparty/tiny-cuda-nn)
else()
    message(STATUS "tiny-cuda-nn: DISABLED (set -DUSE_TINY_CUDA_NN=ON to enable)")
endif()

# =============================================================================
# Third-party dependencies
# =============================================================================

# raylib
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/raylib)

# =============================================================================
# Main application
# =============================================================================

add_subdirectory(src)

# Set Flomo as the startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Flomo)
