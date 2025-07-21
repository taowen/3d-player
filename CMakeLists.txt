cmake_minimum_required(VERSION 3.20)

project(integration-test 
    VERSION 1.0.0
    DESCRIPTION "IW3 C++ Hello World"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# AddressSanitizer temporarily disabled for NPU testing
# message(STATUS "AddressSanitizer enabled globally")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fsanitize=address")

# Include AddressSanitizer support
include(cmake/AddressSanitizer.cmake)

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
    src/hw_video_decoder.cpp
    src/audio_decoder.cpp
    src/rgb_video_decoder.cpp
    src/video_player.cpp
    src/audio_player.cpp
    src/audio_video_player.cpp
    tests/test_mkv_stream_reader.cpp
    tests/test_hw_video_decoder.cpp
    tests/test_audio_decoder.cpp
    tests/test_rgb_video_decoder.cpp
    tests/test_video_player.cpp
    tests/test_audio_player.cpp
    tests/test_audio_video_player.cpp
    tests/test_tensorrt.cpp
)

# Link Catch2 to the executable
target_link_libraries(integration-test PRIVATE Catch2::Catch2WithMain)

# Apply Windows/MSVC compile options
target_compile_options(integration-test PRIVATE
    /W4 /WX- /permissive- /std:c++latest
    /Zi                 # Generate debug info for better error reports
    /utf-8              # Use UTF-8 encoding for source files
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
    winmm.lib
)

# Link FFmpeg to target
include(cmake/FindFFmpeg.cmake)
target_link_ffmpeg(integration-test)
copy_ffmpeg_dlls(integration-test)

# 添加 CUDA 支持
find_package(CUDAToolkit REQUIRED)

# 添加 TensorRT 支持
include(cmake/FindTensorRT.cmake)
target_link_tensorrt(integration-test)
target_link_libraries(integration-test PRIVATE CUDA::cudart)
copy_tensorrt_dlls(integration-test)

# Setup AddressSanitizer for integration-test
setup_asan(integration-test)


# 添加 3d_player 可执行文件
add_executable(3d_player 
    src/main.cpp
    src/mkv_stream_reader.cpp
    src/hw_video_decoder.cpp
    src/audio_decoder.cpp
    src/rgb_video_decoder.cpp
    src/video_player.cpp
    src/audio_player.cpp
    src/audio_video_player.cpp
)

# 应用编译选项
target_compile_options(3d_player PRIVATE
    /W4 /WX- /permissive- /std:c++latest
    /Zi                 # Generate debug info for better error reports
    /utf-8              # Use UTF-8 encoding for source files
)

# 链接 DirectX 和 Windows 系统库
target_link_libraries(3d_player PRIVATE
    d3d11.lib
    dxgi.lib
    d3dcompiler.lib
    ole32.lib
    oleaut32.lib
    user32.lib          # Win32 API
    gdi32.lib           # GDI API
    kernel32.lib        # Windows Kernel API
    winmm.lib           # Windows Multimedia API
)

# 链接 FFmpeg
target_link_ffmpeg(3d_player)
copy_ffmpeg_dlls(3d_player)

# 链接 TensorRT
target_link_tensorrt(3d_player)
target_link_libraries(3d_player PRIVATE CUDA::cudart)
copy_tensorrt_dlls(3d_player)

# Setup AddressSanitizer for 3d_player
setup_asan(3d_player)