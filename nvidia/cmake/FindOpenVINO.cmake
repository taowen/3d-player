# FindOpenVINO.cmake - Find OpenVINO libraries and setup targets

# 设置 OpenVINO 路径
set(OPENVINO_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/openvino_toolkit_windows_2025.2.0.19140.c01cd93e24d_x86_64")
set(OPENVINO_CMAKE_DIR "${OPENVINO_ROOT_DIR}/runtime/cmake")

# 根据构建类型选择正确的 OpenVINO 库路径
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPENVINO_BIN_DIR "${OPENVINO_ROOT_DIR}/runtime/bin/intel64/Debug")
else()
    set(OPENVINO_BIN_DIR "${OPENVINO_ROOT_DIR}/runtime/bin/intel64/Release")
endif()

set(OPENVINO_TBB_BIN_DIR "${OPENVINO_ROOT_DIR}/runtime/3rdparty/tbb/bin")

# 验证 OpenVINO 目录存在
if(NOT EXISTS ${OPENVINO_CMAKE_DIR})
    message(FATAL_ERROR "OpenVINO CMake directory not found: ${OPENVINO_CMAKE_DIR}")
endif()

if(NOT EXISTS ${OPENVINO_BIN_DIR})
    message(FATAL_ERROR "OpenVINO bin directory not found: ${OPENVINO_BIN_DIR}")
endif()

# 设置 OpenVINO_DIR 以便 find_package 能找到
set(OpenVINO_DIR ${OPENVINO_CMAKE_DIR})

message(STATUS "Found OpenVINO:")
message(STATUS "  Root dir: ${OPENVINO_ROOT_DIR}")
message(STATUS "  CMake dir: ${OPENVINO_CMAKE_DIR}")
message(STATUS "  Bin dir: ${OPENVINO_BIN_DIR}")

# 查找 OpenVINO 包
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Function to copy OpenVINO DLLs to target directory
function(copy_openvino_dlls target_name)
    # 使用生成器表达式根据构建配置选择正确的路径
    set(OPENVINO_BIN_DEBUG "${OPENVINO_ROOT_DIR}/runtime/bin/intel64/Debug")
    set(OPENVINO_BIN_RELEASE "${OPENVINO_ROOT_DIR}/runtime/bin/intel64/Release")
    
    # Copy OpenVINO core DLLs - 使用条件复制
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$<IF:$<CONFIG:Debug>,${OPENVINO_BIN_DEBUG},${OPENVINO_BIN_RELEASE}>"
        $<TARGET_FILE_DIR:${target_name}>
        COMMENT "Copying OpenVINO DLLs to output directory"
    )
    
    # Copy TBB DLLs (required by OpenVINO)
    if(EXISTS ${OPENVINO_TBB_BIN_DIR})
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${OPENVINO_TBB_BIN_DIR}
            $<TARGET_FILE_DIR:${target_name}>
            COMMENT "Copying TBB DLLs to output directory"
        )
        message(STATUS "Will copy TBB DLLs from: ${OPENVINO_TBB_BIN_DIR}")
    else()
        message(WARNING "TBB bin directory not found: ${OPENVINO_TBB_BIN_DIR}")
    endif()
    
    message(STATUS "Will copy OpenVINO DLLs from Debug or Release directory based on configuration")
endfunction() 