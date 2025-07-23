#include <catch2/catch_test_macros.hpp>
#include "../src/float_rgb_video_decoder.h"
#include "../src/rgb_video_decoder.h"
#include "../src/hw_video_decoder.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <d3d11.h>
#include <wrl/client.h>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}

using Microsoft::WRL::ComPtr;

const std::string TEST_FILE = "test_data/sample_hw.mkv";

// 辅助函数：从 D3D11 浮点纹理读取像素数据
std::vector<float> readD3D11FloatTexturePixels(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture) {
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
        std::cerr << "Failed to create staging texture, HRESULT: 0x" << std::hex << hr << std::endl;
        return std::vector<float>();
    }
    
    // 复制纹理到 staging
    context->CopyResource(staging_texture.Get(), texture);
    
    // 映射并读取数据
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture, HRESULT: 0x" << std::hex << hr << std::endl;
        return std::vector<float>();
    }
    
    std::vector<float> pixels;
    size_t pixel_count = desc.Width * desc.Height * 4; // RGBA
    pixels.resize(pixel_count);
    
    // DXGI_FORMAT_R32G32B32A32_FLOAT 每个像素16字节
    for (size_t y = 0; y < desc.Height; ++y) {
        const float* row_src = reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(mapped.pData) + y * mapped.RowPitch);
        float* row_dst = pixels.data() + y * desc.Width * 4;
        memcpy(row_dst, row_src, desc.Width * 4 * sizeof(float));
    }
    
    context->Unmap(staging_texture.Get(), 0);
    
    return pixels;
}

TEST_CASE("FloatRgbVideoDecoder basic functionality", "[test_float_rgb_video_decoder.cpp]") {
    // 检查测试文件是否存在
    std::ifstream file(TEST_FILE);
    REQUIRE(file.good());
    file.close();
    
    FloatRgbVideoDecoder decoder;
    
    SECTION("Verify float pixel values are in correct range") {
        REQUIRE(decoder.open(TEST_FILE));
        
        FloatRgbVideoDecoder::DecodedFloatRgbFrame frame;
        bool success = decoder.readNextFrame(frame);
        REQUIRE(success);
        REQUIRE(frame.is_valid);
        
        // 获取D3D11设备和上下文
        auto rgb_decoder = decoder.getRgbDecoder();
        REQUIRE(rgb_decoder != nullptr);
        auto hw_decoder = rgb_decoder->getHwDecoder();
        REQUIRE(hw_decoder != nullptr);
        
        auto device = hw_decoder->getD3D11Device();
        REQUIRE(device != nullptr);
        
        ComPtr<ID3D11DeviceContext> context;
        device->GetImmediateContext(&context);
        REQUIRE(context != nullptr);
        
        // 读取浮点纹理像素数据
        auto pixels = readD3D11FloatTexturePixels(device, context.Get(), frame.float_texture.Get());
        REQUIRE_FALSE(pixels.empty());
        
        // 验证像素值在[0,1]范围内
        bool has_valid_pixels = false;
        int sample_count = 0;
        const int max_samples = 1000;
        
        for (size_t i = 0; i < pixels.size() && sample_count < max_samples; i += 4) {
            float r = pixels[i];
            float g = pixels[i + 1];
            float b = pixels[i + 2];
            float a = pixels[i + 3];
            
            // 检查RGBA值都在[0,1]范围内
            REQUIRE(r >= 0.0f);
            REQUIRE(r <= 1.0f);
            REQUIRE(g >= 0.0f);
            REQUIRE(g <= 1.0f);
            REQUIRE(b >= 0.0f);
            REQUIRE(b <= 1.0f);
            REQUIRE(a >= 0.0f);
            REQUIRE(a <= 1.0f);
            
            // 检查是否有非零像素（避免全黑图像）
            if (r > 0.01f || g > 0.01f || b > 0.01f) {
                has_valid_pixels = true;
            }
            
            sample_count++;
        }
        
        REQUIRE(has_valid_pixels);
        std::cout << "Float pixel values are in correct [0,1] range" << std::endl;
        
        // 释放AVFrame内存
        av_frame_free(&frame.rgb_frame.hw_frame.frame);
    }
}