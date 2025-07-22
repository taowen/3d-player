#include <catch2/catch_test_macros.hpp>
#include "../src/audio_video_player.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <d3d11.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

const std::string TEST_FILE = "test_data/sample_hw.mkv";
const std::string AUDIO_TEST_FILE = "test_data/sample_with_audio.mkv";


// 辅助函数声明（实现在 test_video_player.cpp 中）
extern std::vector<uint8_t> readTexturePixels(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture);
extern bool isValidPixelData(const std::vector<uint8_t>& pixels, int width, int height);


TEST_CASE("AudioVideoPlayer basic functionality", "[audio_video_player][test_audio_video_player.cpp]") {
    AudioVideoPlayer player;
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(player.open("nonexistent_file.mkv"));
        REQUIRE_FALSE(player.isOpen());
    }
    
    SECTION("Open valid test file without audio and simulate time-based playback") {
        std::ifstream file(TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << TEST_FILE << " not found");
        }
        
        // 第一步：打开视频文件（获取设备）
        REQUIRE(player.open(TEST_FILE));
        REQUIRE(player.isOpen());
        REQUIRE_FALSE(player.isEOF());
        
        // 从 AudioVideoPlayer 获取 D3D11 设备
        ID3D11Device* device = player.getD3D11Device();
        ID3D11DeviceContext* context = player.getD3D11DeviceContext();
        REQUIRE(device != nullptr);
        REQUIRE(context != nullptr);
        
        // 获取视频尺寸
        VideoPlayer* video_player = player.getVideoPlayer();
        REQUIRE(video_player != nullptr);
        int width = video_player->getWidth();
        int height = video_player->getHeight();
        REQUIRE(width > 0);
        REQUIRE(height > 0);
        
        // 创建测试用的渲染目标纹理
        D3D11_TEXTURE2D_DESC render_target_desc = {};
        render_target_desc.Width = width;
        render_target_desc.Height = height;
        render_target_desc.MipLevels = 1;
        render_target_desc.ArraySize = 1;
        render_target_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        render_target_desc.SampleDesc.Count = 1;
        render_target_desc.SampleDesc.Quality = 0;
        render_target_desc.Usage = D3D11_USAGE_DEFAULT;
        render_target_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        render_target_desc.CPUAccessFlags = 0;
        render_target_desc.MiscFlags = 0;
        
        ComPtr<ID3D11Texture2D> render_target;
        HRESULT hr = device->CreateTexture2D(&render_target_desc, nullptr, &render_target);
        REQUIRE(SUCCEEDED(hr));
        
        // 第二步：设置渲染目标
        REQUIRE(player.setRenderTarget(render_target));
        REQUIRE(player.isVideoReady());
        
        // 模拟时间驱动的播放
        double current_time = 0.0;
        double time_step = 1.0 / 30.0;  // 30 FPS 播放
        int frame_count = 0;
        int max_test_frames = 5;
        
        std::vector<std::vector<uint8_t>> frame_pixels;
        
        while (frame_count < max_test_frames && !player.isEOF()) {
            // 调用 onTimer 进行内部渲染
            player.onTimer(current_time);
            
            // 读取渲染目标纹理像素数据
            std::vector<uint8_t> pixels = readTexturePixels(device, context, render_target.Get());
            
            // 如果像素数据有效，说明发生了渲染
            int texture_width = video_player->getWidth();
            int texture_height = video_player->getHeight();
            if (!pixels.empty() && isValidPixelData(pixels, width, height)) {
                // 检查是否是新的一帧（与之前的帧不同）
                bool is_new_frame = frame_pixels.empty();
                if (!is_new_frame && !frame_pixels.empty()) {
                    const auto& last_frame = frame_pixels.back();
                    size_t min_size = (pixels.size() < last_frame.size()) ? pixels.size() : last_frame.size();
                    for (size_t i = 0; i < min_size; i += 4) {  // BGRA format
                        if (pixels[i] != last_frame[i] || pixels[i + 1] != last_frame[i + 1] || pixels[i + 2] != last_frame[i + 2]) {
                            is_new_frame = true;
                            break;
                        }
                    }
                }
                
                if (is_new_frame) {
                    // 保存像素数据用于对比
                    frame_pixels.push_back(pixels);
                    frame_count++;
                    
                    std::cout << "Frame " << frame_count << " at time " << current_time << "s" << std::endl;
                }
            }
            
            // 推进时间
            current_time += time_step;
        }
        
        // 验证我们至少获得了一些帧
        REQUIRE(frame_count > 0);
        REQUIRE(frame_pixels.size() > 0);
        
        // 验证不同帧之间的像素数据是不同的（如果有多帧）
        if (frame_pixels.size() > 1) {
            // 比较前两帧，应该有差异
            const auto& frame1 = frame_pixels[0];
            const auto& frame2 = frame_pixels[1];
            
            bool frames_different = false;
            size_t min_size = (frame1.size() < frame2.size()) ? frame1.size() : frame2.size();
            for (size_t i = 0; i < min_size; i += 4) {  // BGRA format
                if (frame1[i] != frame2[i] || frame1[i + 1] != frame2[i + 1] || frame1[i + 2] != frame2[i + 2]) {
                    frames_different = true;
                    break;
                }
            }
            
            REQUIRE(frames_different);  // 不同帧应该有不同的像素数据
        }
        
        player.close();
        REQUIRE_FALSE(player.isOpen());
    }
}


TEST_CASE("AudioVideoPlayer with audio functionality", "[audio_video_player][test_audio_video_player.cpp]") {
    AudioVideoPlayer player;
    
    SECTION("Open file with audio and test audio initialization") {
        std::ifstream file(AUDIO_TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << AUDIO_TEST_FILE << " not found");
        }
        
        // 第一步：打开音视频文件
        REQUIRE(player.open(AUDIO_TEST_FILE));
        REQUIRE(player.isOpen());
        
        // 检查是否有音频流
        if (player.hasAudio()) {
            std::cout << "Audio stream detected" << std::endl;
            
            // 获取音频解码器
            AudioDecoder* audio_decoder = player.getAudioDecoder();
            REQUIRE(audio_decoder != nullptr);
            
            // 获取设备并创建渲染目标
            ID3D11Device* device = player.getD3D11Device();
            ID3D11DeviceContext* context = player.getD3D11DeviceContext();
            REQUIRE(device != nullptr);
            REQUIRE(context != nullptr);
            
            // 获取视频尺寸
            VideoPlayer* video_player_inner = player.getVideoPlayer();
            REQUIRE(video_player_inner != nullptr);
            int video_width = video_player_inner->getWidth();
            int video_height = video_player_inner->getHeight();
            REQUIRE(video_width > 0);
            REQUIRE(video_height > 0);
            
            // 创建测试用的渲染目标纹理
            D3D11_TEXTURE2D_DESC render_target_desc = {};
            render_target_desc.Width = video_width;
            render_target_desc.Height = video_height;
            render_target_desc.MipLevels = 1;
            render_target_desc.ArraySize = 1;
            render_target_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            render_target_desc.SampleDesc.Count = 1;
            render_target_desc.SampleDesc.Quality = 0;
            render_target_desc.Usage = D3D11_USAGE_DEFAULT;
            render_target_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            render_target_desc.CPUAccessFlags = 0;
            render_target_desc.MiscFlags = 0;
            
            ComPtr<ID3D11Texture2D> render_target;
            HRESULT hr = device->CreateTexture2D(&render_target_desc, nullptr, &render_target);
            REQUIRE(SUCCEEDED(hr));
            
            // 设置渲染目标
            REQUIRE(player.setRenderTarget(render_target));
            REQUIRE(player.isVideoReady());
            
            // 初始化音频播放器
            bool audio_initialized = player.initializeAudio();
            if (audio_initialized) {
                std::cout << "Audio initialization successful" << std::endl;
                REQUIRE(player.isAudioReady());
                
                // 测试音频缓冲区状态
                UINT32 total_frames, padding_frames, available_frames;
                bool status_ok = player.getAudioBufferStatus(total_frames, padding_frames, available_frames);
                REQUIRE(status_ok);
                REQUIRE(total_frames > 0);
                
                std::cout << "Audio buffer status: " << total_frames << " total, " 
                         << padding_frames << " padding, " << available_frames << " available" << std::endl;
            } else {
                std::cout << "Audio initialization failed" << std::endl;
            }
            
            // 模拟短时间播放
            double current_time = 0.0;
            double time_step = 1.0 / 30.0;  // 30 FPS 播放
            int frame_count = 0;
            int max_test_frames = 3;
            
            while (frame_count < max_test_frames && !player.isEOF()) {
                // 调用 onTimer 进行音视频播放
                player.onTimer(current_time);
                
                // 读取渲染目标纹理像素数据
                std::vector<uint8_t> pixels = readTexturePixels(device, context, render_target.Get());
                
                // 如果像素数据有效，说明发生了渲染
                int frame_width = video_player_inner->getWidth();
                int frame_height = video_player_inner->getHeight();
                if (!pixels.empty() && isValidPixelData(pixels, frame_width, frame_height)) {
                    frame_count++;
                    std::cout << "Frame " << frame_count << " at time " << current_time << "s" << std::endl;
                }
                
                // 推进时间
                current_time += time_step;
            }
            
            // 验证我们至少获得了一些帧
            REQUIRE(frame_count > 0);
            
        } else {
            std::cout << "No audio stream detected" << std::endl;
        }
        
        player.close();
        REQUIRE_FALSE(player.isOpen());
    }
}


TEST_CASE("AudioVideoPlayer component access", "[audio_video_player][test_audio_video_player.cpp]") {
    AudioVideoPlayer player;
    
    SECTION("Test component access methods") {
        std::ifstream file(TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << TEST_FILE << " not found");
        }
        
        // 打开文件
        REQUIRE(player.open(TEST_FILE));
        REQUIRE(player.isOpen());
        
        // 测试组件访问
        VideoPlayer* video_player = player.getVideoPlayer();
        REQUIRE(video_player != nullptr);
        
        AudioPlayer* audio_player = player.getAudioPlayer();
        REQUIRE(audio_player != nullptr);
        
        // 测试解码器访问
        StereoVideoDecoder* stereo_decoder = player.getStereoDecoder();
        REQUIRE(stereo_decoder != nullptr);
        
        // 测试 D3D11 设备访问
        ID3D11Device* device = player.getD3D11Device();
        ID3D11DeviceContext* context = player.getD3D11DeviceContext();
        REQUIRE(device != nullptr);
        REQUIRE(context != nullptr);
        
        player.close();
    }
} 