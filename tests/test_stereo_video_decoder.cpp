#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "../src/stereo_video_decoder.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <vector>
#include <wrl/client.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/rational.h>
}

using Microsoft::WRL::ComPtr;

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
static void saveRGBTextureToBMP(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture, const std::string& filename) {
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

// 辅助函数：从 D3D11 浮点纹理读取像素数据
static std::vector<float> readD3D11FloatTexturePixels(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture) {
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
    
    // 复制像素数据
    std::vector<float> pixels(desc.Width * desc.Height * 4);
    const float* src_data = reinterpret_cast<const float*>(mapped.pData);
    
    for (uint32_t y = 0; y < desc.Height; ++y) {
        const float* src_row = reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(mapped.pData) + y * mapped.RowPitch);
        
        for (uint32_t x = 0; x < desc.Width; ++x) {
            size_t dst_idx = (y * desc.Width + x) * 4;
            pixels[dst_idx + 0] = src_row[x * 4 + 0]; // R
            pixels[dst_idx + 1] = src_row[x * 4 + 1]; // G
            pixels[dst_idx + 2] = src_row[x * 4 + 2]; // B
            pixels[dst_idx + 3] = src_row[x * 4 + 3]; // A
        }
    }
    
    context->Unmap(staging_texture.Get(), 0);
    return pixels;
}

// 辅助函数：将浮点纹理保存为BMP文件
static void saveFloatTextureToBMP(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture, const std::string& filename) {
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

TEST_CASE("Stereo Video Decoder Frame Reading", "[stereo_video_decoder][frames][test_stereo_video_decoder.cpp]") {
    SECTION("Read frames from valid video") {
        StereoVideoDecoder decoder;
        
        std::string test_video = "test_data/sample_hw.mkv";
        
        if (!decoder.open(test_video)) {
            WARN("Test video not found: " + test_video + " - skipping frame reading tests");
            return;
        }
        
        REQUIRE(decoder.isOpen());
        
        int frame_count = 0;
        const int max_frames = 10;
        
        std::cout << "Reading up to " << max_frames << " frames..." << std::endl;
        
        while (frame_count < max_frames && !decoder.isEOF()) {
            DecodedStereoFrame frame;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            bool success = decoder.readNextFrame(frame);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (!success) {
                if (decoder.isEOF()) {
                    std::cout << "Reached end of file at frame " << frame_count << std::endl;
                    break;
                } else {
                    FAIL("Failed to read frame " + std::to_string(frame_count));
                    break;
                }
            }
            
            REQUIRE(frame.is_valid);
            REQUIRE(frame.stereo_texture != nullptr);
            
            // 计算PTS时间戳
            double pts_seconds = 0.0;
            if (frame.frame && frame.frame->pts != AV_NOPTS_VALUE) {
                AVRational time_base = decoder.getVideoTimeBase();
                pts_seconds = frame.frame->pts * av_q2d(time_base);
            }
            REQUIRE(pts_seconds >= 0.0);
            
            D3D11_TEXTURE2D_DESC texture_desc;
            frame.stereo_texture->GetDesc(&texture_desc);
            
            REQUIRE(texture_desc.Width == static_cast<UINT>(decoder.getWidth()));
            REQUIRE(texture_desc.Height == static_cast<UINT>(decoder.getHeight()));
            REQUIRE(texture_desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT);
            
            // 验证像素数据有效性
            ID3D11Device* device = nullptr;
            frame.stereo_texture->GetDevice(&device);
            REQUIRE(device != nullptr);
            
            ID3D11DeviceContext* context = nullptr;
            device->GetImmediateContext(&context);
            REQUIRE(context != nullptr);
            
            // 创建staging纹理来读取像素数据
            D3D11_TEXTURE2D_DESC staging_desc = texture_desc;
            staging_desc.Usage = D3D11_USAGE_STAGING;
            staging_desc.BindFlags = 0;
            staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            staging_desc.MiscFlags = 0;
            
            ID3D11Texture2D* staging_texture = nullptr;
            HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
            REQUIRE(SUCCEEDED(hr));
            REQUIRE(staging_texture != nullptr);
            
            // 复制纹理数据到staging纹理
            context->CopyResource(staging_texture, frame.stereo_texture.Get());
            
            // 映射staging纹理并读取像素数据
            D3D11_MAPPED_SUBRESOURCE mapped_resource;
            hr = context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped_resource);
            REQUIRE(SUCCEEDED(hr));
            
            // 验证像素数据
            const float* pixel_data = static_cast<const float*>(mapped_resource.pData);
            REQUIRE(pixel_data != nullptr);
            
            // 检查前几个像素值，确保不全为零且在合理范围内
            bool has_non_zero_pixel = false;
            bool has_valid_range_pixel = false;
            const int sample_pixels = (std::min)(100, static_cast<int>(texture_desc.Width * texture_desc.Height));
            
            for (int i = 0; i < sample_pixels; i++) {
                float r = pixel_data[i * 4 + 0];
                float g = pixel_data[i * 4 + 1]; 
                float b = pixel_data[i * 4 + 2];
                float a = pixel_data[i * 4 + 3];
                
                // 检查是否有非零像素
                if (r != 0.0f || g != 0.0f || b != 0.0f || a != 0.0f) {
                    has_non_zero_pixel = true;
                }
                
                // 检查像素值是否在合理范围内 (0.0 到 1.0)
                if ((r >= 0.0f && r <= 1.0f) && (g >= 0.0f && g <= 1.0f) && 
                    (b >= 0.0f && b <= 1.0f) && (a >= 0.0f && a <= 1.0f)) {
                    has_valid_range_pixel = true;
                }
            }
            
            context->Unmap(staging_texture, 0);
            staging_texture->Release();
            context->Release();
            device->Release();
            
            REQUIRE(has_non_zero_pixel);
            REQUIRE(has_valid_range_pixel);
            
            // 保存前3帧为BMP文件用于肉眼比对
            if (frame_count < 3) {
                ID3D11Device* save_device = nullptr;
                frame.stereo_texture->GetDevice(&save_device);
                
                ID3D11DeviceContext* save_context = nullptr;
                save_device->GetImmediateContext(&save_context);
                
                // 保存输入RGB帧
                if (frame.input_frame && frame.input_frame->rgb_frame.rgb_texture) {
                    std::string input_filename = "input_frame_" + std::to_string(frame_count) + ".bmp";
                    saveRGBTextureToBMP(save_device, save_context, 
                                      frame.input_frame->rgb_frame.rgb_texture.Get(), 
                                      input_filename);
                    std::cout << "Saved input frame: " << input_filename << std::endl;
                }
                
                // 保存立体视觉输出帧
                std::string output_filename = "stereo_output_frame_" + std::to_string(frame_count) + ".bmp";
                saveFloatTextureToBMP(save_device, save_context, 
                                    frame.stereo_texture.Get(), 
                                    output_filename);
                std::cout << "Saved stereo output frame: " << output_filename << std::endl;
                
                save_context->Release();
                save_device->Release();
            }
            
            std::cout << "Frame " << frame_count 
                      << ": " << texture_desc.Width << "x" << texture_desc.Height
                      << ", PTS: " << pts_seconds << "s"
                      << ", Processing time: " << duration.count() << "ms"
                      << ", Pixel validation: PASSED" << std::endl;
            
            // 释放AVFrame内存
            if (frame.frame) {
                av_frame_free(&frame.frame);
            }
            
            frame_count++;
        }
        
        REQUIRE(frame_count > 0);
        std::cout << "Successfully processed " << frame_count << " frames" << std::endl;
        
        decoder.close();
    }
}