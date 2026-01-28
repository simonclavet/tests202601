# Flomo - BVH/FBX Animation Viewer
# Root CMakeLists.txt - Project configuration and subdirectories

cmake_minimum_required(VERSION 3.21)

# MSVC runtime configuration (before project())
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
endif()

# Only Debug and RelWithDebInfo configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo" CACHE STRING "" FORCE)

project(FlomoProject LANGUAGES C CXX CUDA)

# =============================================================================
# Global settings
# =============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# =============================================================================
# Libtorch paths (shared between targets)
# =============================================================================

set(TORCH_DEBUG_PATH "${CMAKE_SOURCE_DIR}/thirdparty/libtorch/debug")
set(TORCH_RELEASE_PATH "${CMAKE_SOURCE_DIR}/thirdparty/libtorch/release")
set(TORCH_INCLUDE_DIR "${TORCH_RELEASE_PATH}/include")
set(TORCH_INCLUDE_DIR_2 "${TORCH_RELEASE_PATH}/include/torch/csrc/api/include")

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
