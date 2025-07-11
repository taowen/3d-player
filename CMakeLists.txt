cmake_minimum_required(VERSION 3.20)

project(integration-test 
    VERSION 1.0.0
    DESCRIPTION "IW3 C++ Hello World"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(integration-test 
    src/mkv_stream_reader.cpp
    tests/test_mkv_stream_reader.cpp
)

# Link Catch2 to the executable
target_link_libraries(integration-test PRIVATE Catch2::Catch2WithMain)

# Apply Windows/MSVC compile options
target_compile_options(integration-test PRIVATE
    /W4 /WX- /permissive- /std:c++latest
)

# Set Visual Studio startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT integration-test)

# Link DirectX libraries (Windows only)
target_link_libraries(integration-test PRIVATE
    d3d11.lib
    dxgi.lib
    d3dcompiler.lib
    ole32.lib
    oleaut32.lib
)

# Link FFmpeg to target
include(cmake/FindFFmpeg.cmake)
target_link_ffmpeg(integration-test)
copy_ffmpeg_dlls(integration-test)

# Link CUDA to target (runtime only, no nvcc compilation)
include(cmake/FindCUDA.cmake)
target_link_cuda(integration-test)
copy_cuda_dlls(integration-test)

# Link TensorRT to target
include(cmake/FindTensorRT.cmake)
target_link_tensorrt(integration-test)
copy_tensorrt_dlls(integration-test)