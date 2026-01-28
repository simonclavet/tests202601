# FBX2BVH - FBX to BVH converter using raylib and ufbx

set(FBX2BVH_SOURCES
    fbx_converter.h
    main.cpp
    fbx_converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/ufbx/ufbx.c
)

add_executable(FBX2BVH ${FBX2BVH_SOURCES})

target_include_directories(FBX2BVH PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/raylib/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/ufbx
)

target_link_libraries(FBX2BVH PRIVATE raylib)

if(WIN32)
    target_link_libraries(FBX2BVH PRIVATE opengl32 gdi32 winmm)
endif()

target_compile_definitions(FBX2BVH PRIVATE
    _CRT_SECURE_NO_WARNINGS
)

if(MSVC)
    target_compile_options(FBX2BVH PRIVATE
        /W4 /WX
        /wd4100  # unreferenced formal parameter
        /wd4267  # size_t to int
        /wd4244  # double to float
        /wd4456  # declaration hides previous local
        /wd4457  # declaration hides function parameter
    )
endif()

set_target_properties(FBX2BVH PROPERTIES
    OUTPUT_NAME "fbx2bvh"
)
