# AddressSanitizer.cmake - AddressSanitizer support for Windows/MSVC

# Function to copy AddressSanitizer DLLs to target directory
function(copy_asan_dlls target_name)
    
    if(MSVC)
        # Find Visual Studio installation path
        get_filename_component(COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
        
        # Try different possible paths for ASAN DLLs
        set(ASAN_DLL_PATHS
            "${COMPILER_DIR}"
            "${COMPILER_DIR}/../../../bin/Hostx64/x64"
            "${COMPILER_DIR}/../bin/Hostx64/x64"
            "$ENV{VCToolsRedistDir}/x64/Microsoft.VC143.CRT"
            "$ENV{VCINSTALLDIR}/Redist/MSVC/*/x64/Microsoft.VC143.ASAN"
        )
        
        # Common ASAN DLL names
        set(ASAN_DLL_NAMES
            "clang_rt.asan_dynamic-x86_64.dll"
            "clang_rt.asan_dbg_dynamic-x86_64.dll"
        )
        
        set(FOUND_ASAN_DLLS "")
        
        foreach(dll_path ${ASAN_DLL_PATHS})
            if(EXISTS "${dll_path}")
                foreach(dll_name ${ASAN_DLL_NAMES})
                    file(GLOB found_dll "${dll_path}/${dll_name}")
                    if(found_dll)
                        list(APPEND FOUND_ASAN_DLLS ${found_dll})
                    endif()
                endforeach()
            endif()
        endforeach()
        
        if(FOUND_ASAN_DLLS)
            list(REMOVE_DUPLICATES FOUND_ASAN_DLLS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FOUND_ASAN_DLLS}
                $<TARGET_FILE_DIR:${target_name}>
                COMMENT "Copying AddressSanitizer DLLs to output directory"
            )
            message(STATUS "Will copy AddressSanitizer DLLs: ${FOUND_ASAN_DLLS}")
        else()
            message(WARNING "AddressSanitizer enabled but runtime DLLs not found. The executable may fail to run.")
        endif()
    endif()
endfunction()

# Function to configure AddressSanitizer for a target
function(setup_asan target_name)
    # Note: Compile options are set globally, we only need to set link options and copy DLLs
    target_link_options(${target_name} PRIVATE /fsanitize=address)
    copy_asan_dlls(${target_name})
endfunction() 