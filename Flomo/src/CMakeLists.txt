
# Where we keep the ImGui library files for our UI
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/thirdparty/imgui)

# List all the ImGui source files we need to compile
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
)

# The bridge library that lets ImGui work with Raylib
set(RLIMGUI_DIR ${CMAKE_SOURCE_DIR}/thirdparty/rlimgui)
set(RLIMGUI_SOURCES
    ${RLIMGUI_DIR}/rlImGui.cpp
)

# Combine all our source files into one list
set(FLOMO_SOURCES
    flomo.cpp
    cuda_kernels.cu
    ${IMGUI_SOURCES}
    ${RLIMGUI_SOURCES}
)

# All the header files our project uses
set(FLOMO_HEADERS
    pch.h
    math_utils.h
    utils.h
    camera.h
    capsule_data.h
    mesh_utils.h
    geometry_utils.h
    gui_window_file_dialog.h
    profiler.h
    transform_data.h
    bvh_parser.h
    fbx_loader.h
    geno_renderer.h
    app_config.h
    balltree.h
    leg_ik.h
    character_data.h
    anim_database.h
    controlled_character.h
    motion_matching.h
    ${CMAKE_SOURCE_DIR}/thirdparty/ufbx/ufbx.h
    ${IMGUI_DIR}/imgui.h
    ${RLIMGUI_DIR}/rlImGui.h
)

# Actually create the executable from our source files
add_executable(Flomo
    ${FLOMO_SOURCES}
    ${FLOMO_HEADERS}
    ${CMAKE_SOURCE_DIR}/thirdparty/ufbx/ufbx.c
)

# Tell the compiler where to look for include files
target_include_directories(Flomo PRIVATE
    ${CMAKE_SOURCE_DIR}/thirdparty/raylib/src
    ${CMAKE_SOURCE_DIR}/thirdparty/raygui/src
    ${CMAKE_SOURCE_DIR}/thirdparty/raygui
    ${CMAKE_SOURCE_DIR}/thirdparty/ufbx
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Mark third-party headers as "system" to suppress their warnings
target_include_directories(Flomo SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/thirdparty/imgui
    ${CMAKE_SOURCE_DIR}/thirdparty/rlimgui
)

# Same for PyTorch headers
target_include_directories(Flomo SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIR_2}
)

# Use different library paths for Debug vs Release builds
target_link_directories(Flomo PRIVATE
    $<$<CONFIG:Debug>:${TORCH_DEBUG_PATH}/lib>
    $<$<CONFIG:RelWithDebInfo>:${TORCH_RELEASE_PATH}/lib>
)

# Link against all the libraries we need
target_link_libraries(Flomo PRIVATE
    raylib
    torch
    torch_cpu
    torch_cuda
    c10
    c10_cuda
)

# Windows needs some extra system libraries
if(WIN32)
    target_link_libraries(Flomo PRIVATE opengl32 gdi32 winmm)
endif()

# Some preprocessor definitions to make things work
target_compile_definitions(Flomo PRIVATE
    _CRT_SECURE_NO_WARNINGS
    SOURCE_ROOT_PATH="${CMAKE_SOURCE_DIR}"
    NOMINMAX
)

# Visual Studio specific settings
if(MSVC)
    # Compiler flags for C++ files only
    target_compile_options(Flomo PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            /W4 /WX
            # /wd4100: Unreferenced formal parameter
            /wd4100 
            # /wd4267: Conversion from size_t to smaller type, loss of data
            /wd4267 
            # /wd4018: Signed/unsigned mismatch
            /wd4018 
            # /wd4505: Unreferenced local function removed
            /wd4505
            # /wd4127: Conditional expression is constant
            /wd4127
            /Zc:__cplusplus
            /bigobj
        >
    )

    # Copy PyTorch DLLs after building so the program can run
    add_custom_command(TARGET Flomo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Checking for DLL updates..."
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_SOURCE_DIR}/cmake/copy_torch_dlls.cmake"
            "$<IF:$<CONFIG:Debug>,${TORCH_DEBUG_PATH}/lib,${TORCH_RELEASE_PATH}/lib>"
            "$<TARGET_FILE_DIR:Flomo>"
        COMMENT "Updating Torch DLLs if needed"
        VERBATIM
    )
endif()

# Special flags for CUDA compilation (GPU code)
target_compile_options(Flomo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -lineinfo
    >
)

# CUDA-specific build settings
set_target_properties(Flomo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "flomo"
)

# Use precompiled headers for faster builds
target_precompile_headers(Flomo PRIVATE pch.h)

# Don't use precompiled headers for C files (incompatible)
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/thirdparty/ufbx/ufbx.c
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

# Don't precompile third-party headers and suppress their warnings
set_source_files_properties(
    ${IMGUI_SOURCES}
    ${RLIMGUI_SOURCES}
    PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
        COMPILE_FLAGS "/W0"
)