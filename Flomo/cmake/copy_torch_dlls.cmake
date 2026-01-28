# Script to copy DLLs only if they are newer
if(NOT "${CMAKE_ARGV3}" STREQUAL "" AND NOT "${CMAKE_ARGV4}" STREQUAL "")
    set(SOURCE_DIR "${CMAKE_ARGV3}")
    set(DEST_DIR "${CMAKE_ARGV4}")
    
    # Find all DLLs in source directory
    file(GLOB DLL_FILES "${SOURCE_DIR}/*.dll")
    
    foreach(DLL_FILE ${DLL_FILES})
        get_filename_component(FILENAME "${DLL_FILE}" NAME)
        set(DEST_FILE "${DEST_DIR}/${FILENAME}")
        
        # Check if destination doesn't exist or source is newer
        if(NOT EXISTS "${DEST_FILE}" OR 
           "${DLL_FILE}" IS_NEWER_THAN "${DEST_FILE}")
            message(STATUS "Copying ${FILENAME}")
            file(COPY "${DLL_FILE}" DESTINATION "${DEST_DIR}")
        endif()
    endforeach()
endif()