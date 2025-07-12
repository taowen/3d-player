#include <catch2/catch_test_macros.hpp>
#include "../src/video_player.h"
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


// 辅助函数：从 D3D11 纹理读取像素数据
std::vector<uint8_t> readTexturePixels(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture) {
    if (!texture) {
        return std::vector<uint8_t>();
    }
    
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    // 创建 staging 纹理用于 CPU 读取
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    
    ComPtr<ID3D11Texture2D> staging_texture;
    HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        return std::vector<uint8_t>();
    }
    
    // 复制纹理到 staging
    context->CopyResource(staging_texture.Get(), texture);
    
    // 映射并读取数据
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        return std::vector<uint8_t>();
    }
    
    std::vector<uint8_t> pixels;
    pixels.resize(desc.Height * mapped.RowPitch);
    memcpy(pixels.data(), mapped.pData, pixels.size());
    
    context->Unmap(staging_texture.Get(), 0);
    
    return pixels;
}


// 辅助函数：检查像素数据是否有效（非全零）
bool isValidPixelData(const std::vector<uint8_t>& pixels, int width, int height) {
    if (pixels.empty()) {
        return false;
    }
    
    // 检查至少有一些非零像素
    int non_zero_count = 0;
    for (size_t i = 0; i < pixels.size(); i += 4) {  // BGRA format
        if (pixels[i] != 0 || pixels[i + 1] != 0 || pixels[i + 2] != 0) {
            non_zero_count++;
        }
    }
    
    // 至少要有 1% 的像素是非零的
    int total_pixels = width * height;
    return non_zero_count > total_pixels * 0.01;
}


TEST_CASE("VideoPlayer basic functionality with render to texture", "[video_player]") {
    VideoPlayer player;
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(player.open("nonexistent_file.mkv"));
        REQUIRE_FALSE(player.isOpen());
    }
    
    SECTION("Open valid test file and simulate time-based playback") {
        std::ifstream file(TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << TEST_FILE << " not found");
        }
        
        // 第一步：打开视频文件（获取设备）
        REQUIRE(player.open(TEST_FILE));
        REQUIRE(player.isOpen());
        REQUIRE_FALSE(player.isEOF());
        
        // 从 VideoPlayer 获取 D3D11 设备
        ID3D11Device* device = player.getD3D11Device();
        ID3D11DeviceContext* context = player.getD3D11DeviceContext();
        REQUIRE(device != nullptr);
        REQUIRE(context != nullptr);
        
        // 获取视频尺寸
        RgbVideoDecoder* rgb_decoder = player.getRgbDecoder();
        REQUIRE(rgb_decoder != nullptr);
        
        // 创建测试用的渲染目标纹理
        D3D11_TEXTURE2D_DESC render_target_desc = {};
        render_target_desc.Width = rgb_decoder->getWidth();
        render_target_desc.Height = rgb_decoder->getHeight();
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
        REQUIRE(player.isReady());
        
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
            int width = rgb_decoder->getWidth();
            int height = rgb_decoder->getHeight();
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

TEST_CASE("VideoPlayer audio WASAPI non-blocking write in onTimer", "[video_player][audio]") {
    VideoPlayer player;
    
    // 使用包含音频的测试文件
    const std::string AUDIO_TEST_FILE = "test_data/sample_with_audio.mkv";
    
    std::ifstream file(AUDIO_TEST_FILE);
    if (!file.good()) {
        SKIP("Test file " << AUDIO_TEST_FILE << " not found");
    }
    
    // 第一步：打开视频文件
    REQUIRE(player.open(AUDIO_TEST_FILE));
    REQUIRE(player.isOpen());
    
    // 检查是否有音频流
    REQUIRE(player.hasAudio());
    
    // 获取设备用于创建渲染目标
    ID3D11Device* device = player.getD3D11Device();
    ID3D11DeviceContext* context = player.getD3D11DeviceContext();
    REQUIRE(device != nullptr);
    REQUIRE(context != nullptr);
    
    // 创建渲染目标纹理
    RgbVideoDecoder* rgb_decoder = player.getRgbDecoder();
    REQUIRE(rgb_decoder != nullptr);
    
    D3D11_TEXTURE2D_DESC render_target_desc = {};
    render_target_desc.Width = rgb_decoder->getWidth();
    render_target_desc.Height = rgb_decoder->getHeight();
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
    REQUIRE(player.isReady());
    
    // 第三步：初始化音频播放器
    REQUIRE(player.initializeAudio());
    REQUIRE(player.isAudioReady());
    
    // 获取初始音频缓冲区状态
    UINT32 initial_total_frames, initial_padding_frames, initial_available_frames;
    REQUIRE(player.getAudioBufferStatus(initial_total_frames, initial_padding_frames, initial_available_frames));
    
    // 验证缓冲区初始状态
    REQUIRE(initial_total_frames > 0);
    REQUIRE(initial_padding_frames == 0);  // 初始时应该没有待播放数据
    REQUIRE(initial_available_frames == initial_total_frames);
    
    // 模拟时间驱动的播放，重点验证音频写入
    double current_time = 0.0;
    double time_step = 1.0 / 60.0;  // 60 FPS 调用频率
    int timer_calls = 0;
    int max_timer_calls = 10;
    
    bool audio_data_written = false;
    UINT32 max_padding_seen = 0;
    
    while (timer_calls < max_timer_calls && !player.isEOF()) {
        // 记录 onTimer 调用前的时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 调用 onTimer（包含音频写入）
        player.onTimer(current_time);
        
        // 记录 onTimer 调用后的时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // 验证 onTimer 调用是非阻塞的（应该在很短时间内完成）
        REQUIRE(duration.count() < 10000);  // 应该在 10ms 内完成
        
        // 检查音频缓冲区状态
        UINT32 total_frames, padding_frames, available_frames;
        if (player.getAudioBufferStatus(total_frames, padding_frames, available_frames)) {
            // 验证缓冲区状态的一致性
            REQUIRE(total_frames == initial_total_frames);
            REQUIRE(padding_frames + available_frames == total_frames);
            
            // 如果有音频数据被写入，padding_frames 应该增加
            if (padding_frames > 0) {
                audio_data_written = true;
                max_padding_seen = (std::max)(max_padding_seen, padding_frames);
            }
        }
        
        // 推进时间和计数器
        current_time += time_step;
        timer_calls++;
    }
    
    // 验证音频数据确实被写入了
    REQUIRE(audio_data_written);
    REQUIRE(max_padding_seen > 0);
    
    std::cout << "Audio test results:" << std::endl;
    std::cout << "  - Timer calls: " << timer_calls << std::endl;
    std::cout << "  - Audio data written: " << (audio_data_written ? "Yes" : "No") << std::endl;
    std::cout << "  - Max padding frames seen: " << max_padding_seen << std::endl;
    std::cout << "  - Buffer total frames: " << initial_total_frames << std::endl;
    
    // 清理
    player.close();
    REQUIRE_FALSE(player.isOpen());
} 