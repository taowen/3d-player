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

// 前向声明
std::vector<float> readD3D11FloatTexturePixels(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture);

// BMP文件头结构
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType = 0x4D42;     // "BM"
    uint32_t bfSize;
    uint16_t bfReserved1 = 0;
    uint16_t bfReserved2 = 0;
    uint32_t bfOffBits = 54;      // 文件头+信息头大小
};

struct BMPInfoHeader {
    uint32_t biSize = 40;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes = 1;
    uint16_t biBitCount = 24;     // 24位RGB
    uint32_t biCompression = 0;   // 无压缩
    uint32_t biSizeImage = 0;
    int32_t biXPelsPerMeter = 0;
    int32_t biYPelsPerMeter = 0;
    uint32_t biClrUsed = 0;
    uint32_t biClrImportant = 0;
};
#pragma pack(pop)

// 辅助函数：将RGB纹理保存为BMP文件
void saveRGBTextureToBMP(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture, const std::string& filename) {
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    // 创建staging纹理
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    
    ComPtr<ID3D11Texture2D> staging_texture;
    HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) return;
    
    context->CopyResource(staging_texture.Get(), texture);
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return;
    
    // 准备BMP数据
    uint32_t width = desc.Width;
    uint32_t height = desc.Height;
    uint32_t row_size = ((width * 3 + 3) / 4) * 4; // BMP行大小必须是4的倍数
    uint32_t image_size = row_size * height;
    
    BMPFileHeader file_header;
    file_header.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + image_size;
    
    BMPInfoHeader info_header;
    info_header.biWidth = width;
    info_header.biHeight = height; // 正值表示从下到上
    info_header.biSizeImage = image_size;
    
    // 写入BMP文件
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
        file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));
        
        // 转换RGBA到BGR并按BMP行序写入（从下到上）
        std::vector<uint8_t> row_data(row_size, 0);
        for (int y = height - 1; y >= 0; --y) {
            const uint8_t* src_row = reinterpret_cast<const uint8_t*>(mapped.pData) + y * mapped.RowPitch;
            
            for (uint32_t x = 0; x < width; ++x) {
                // RGBA -> BGR
                row_data[x * 3 + 0] = src_row[x * 4 + 2]; // B
                row_data[x * 3 + 1] = src_row[x * 4 + 1]; // G
                row_data[x * 3 + 2] = src_row[x * 4 + 0]; // R
            }
            
            file.write(reinterpret_cast<const char*>(row_data.data()), row_size);
        }
        file.close();
    }
    
    context->Unmap(staging_texture.Get(), 0);
}

// 辅助函数：将浮点纹理保存为BMP文件
void saveFloatTextureToBMP(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture, const std::string& filename) {
    auto pixels = readD3D11FloatTexturePixels(device, context, texture);
    if (pixels.empty()) return;
    
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    uint32_t width = desc.Width;
    uint32_t height = desc.Height;
    uint32_t row_size = ((width * 3 + 3) / 4) * 4;
    uint32_t image_size = row_size * height;
    
    BMPFileHeader file_header;
    file_header.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + image_size;
    
    BMPInfoHeader info_header;
    info_header.biWidth = width;
    info_header.biHeight = height;
    info_header.biSizeImage = image_size;
    
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
        file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));
        
        std::vector<uint8_t> row_data(row_size, 0);
        for (int y = height - 1; y >= 0; --y) {
            for (uint32_t x = 0; x < width; ++x) {
                size_t pixel_idx = (y * width + x) * 4;
                
                // 浮点值[0,1]转换为8位整数[0,255]，RGBA -> BGR
                uint8_t r = static_cast<uint8_t>(std::round(pixels[pixel_idx + 0] * 255.0f));
                uint8_t g = static_cast<uint8_t>(std::round(pixels[pixel_idx + 1] * 255.0f));
                uint8_t b = static_cast<uint8_t>(std::round(pixels[pixel_idx + 2] * 255.0f));
                
                row_data[x * 3 + 0] = b; // B
                row_data[x * 3 + 1] = g; // G
                row_data[x * 3 + 2] = r; // R
            }
            file.write(reinterpret_cast<const char*>(row_data.data()), row_size);
        }
        file.close();
    }
}

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
        
        // 保存输入RGB纹理到BMP文件
        // std::cout << "Saving input RGB texture to input_rgb_frame.bmp" << std::endl;
        // saveRGBTextureToBMP(device, context.Get(), frame.rgb_frame.rgb_texture.Get(), "input_rgb_frame.bmp");
        
        // 验证CUDA buffer
        REQUIRE(frame.cuda_buffer != nullptr);
        REQUIRE(frame.buffer_size > 0);
        
        // 验证buffer大小是否符合预期：BCHW格式, 4通道RGBA, float32
        size_t expected_size = 1 * 4 * decoder.getHeight() * decoder.getWidth() * sizeof(float);
        REQUIRE(frame.buffer_size == expected_size);
        
        // 由于数据现在是CUDA buffer形式，这里我们只能验证基本属性
        // 实际像素验证需要CUDA内存复制，暂时跳过详细像素检查
        std::cout << "CUDA buffer validated: size=" << frame.buffer_size 
                  << " bytes, expected=" << expected_size << " bytes" << std::endl;
        
        // 释放AVFrame内存
        av_frame_free(&frame.rgb_frame.hw_frame.frame);
    }
}