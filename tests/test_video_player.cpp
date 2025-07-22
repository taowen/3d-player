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
        int width = player.getWidth();
        int height = player.getHeight();
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
            int texture_width = player.getWidth();
            int texture_height = player.getHeight();
            if (!pixels.empty() && isValidPixelData(pixels, texture_width, texture_height)) {
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

// 注意：音频测试已移至 test_audio_video_player.cpp
// VideoPlayer 现在专注于纯视频播放功能 