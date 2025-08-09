cmake_minimum_required(VERSION 3.20)
project(integration-test-intel VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 让 intel 目录能找到根目录 cmake 模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../nvidia/cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fsanitize=address")
include(${CMAKE_CURRENT_SOURCE_DIR}/../nvidia/cmake/AddressSanitizer.cmake)

# Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(Catch2)

# Intel 版本只包含普通音视频播放，不包含立体/ TensorRT / CUDA
add_executable(integration-test-intel
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
)

target_link_libraries(integration-test-intel PRIVATE Catch2::Catch2WithMain)

target_compile_options(integration-test-intel PRIVATE /W4 /WX- /permissive- /std:c++latest /Zi /utf-8)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT integration-test-intel)

# Windows 库
target_link_libraries(integration-test-intel PRIVATE d3d11.lib dxgi.lib d3dcompiler.lib ole32.lib oleaut32.lib winmm.lib)

include(${CMAKE_CURRENT_SOURCE_DIR}/../nvidia/cmake/FindFFmpeg.cmake)
target_link_ffmpeg(integration-test-intel)
copy_ffmpeg_dlls(integration-test-intel)

setup_asan(integration-test-intel)

add_executable(3d_player_intel
    src/main.cpp
    src/mkv_stream_reader.cpp
    src/hw_video_decoder.cpp
    src/audio_decoder.cpp
    src/rgb_video_decoder.cpp
    src/video_player.cpp
    src/audio_player.cpp
    src/audio_video_player.cpp
)

target_compile_options(3d_player_intel PRIVATE /W4 /WX- /permissive- /std:c++latest /Zi /utf-8)

target_link_libraries(3d_player_intel PRIVATE d3d11.lib dxgi.lib d3dcompiler.lib ole32.lib oleaut32.lib user32.lib gdi32.lib kernel32.lib winmm.lib)

target_link_ffmpeg(3d_player_intel)
copy_ffmpeg_dlls(3d_player_intel)

setup_asan(3d_player_intel)

add_executable(audio_only_player
    src/audio_only_player.cpp
    src/mkv_stream_reader.cpp
    src/audio_decoder.cpp
    src/audio_player.cpp
)

target_compile_options(audio_only_player PRIVATE /W4 /WX- /permissive- /std:c++latest /Zi /utf-8)

target_link_libraries(audio_only_player PRIVATE ole32.lib oleaut32.lib user32.lib winmm.lib)

target_link_ffmpeg(audio_only_player)
copy_ffmpeg_dlls(audio_only_player)

setup_asan(audio_only_player)
