#define NOMINMAX  // 防止Windows定义min/max宏
#include <catch2/catch_test_macros.hpp>
#include "../src/float_rgb_video_decoder.h"
#include "../src/rgb_video_decoder.h"
#include "../src/hw_video_decoder.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>  // 用于std::min, std::max
#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_runtime.h>

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
std::vector<float> readCudaBufferPixels(void* cuda_buffer, size_t buffer_size);

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

// 辅助函数：从 D3D11 RGB纹理读取像素数据并转换为浮点格式
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
    
    // 根据纹理格式处理数据
    if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) {
        // UINT8格式：每个像素4字节，需要转换为浮点
        size_t expected_row_bytes = desc.Width * 4; // 4 bytes per pixel
        if (mapped.RowPitch < expected_row_bytes) {
            std::cerr << "Invalid RowPitch: " << mapped.RowPitch 
                      << " < expected: " << expected_row_bytes << std::endl;
            context->Unmap(staging_texture.Get(), 0);
            return std::vector<float>();
        }
        
        for (size_t y = 0; y < desc.Height; ++y) {
            const uint8_t* row_src = reinterpret_cast<const uint8_t*>(mapped.pData) + y * mapped.RowPitch;
            float* row_dst = pixels.data() + y * desc.Width * 4;
            
            for (size_t x = 0; x < desc.Width; ++x) {
                size_t src_idx = x * 4;
                size_t dst_idx = x * 4;
                
                if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM) {
                    // BGRA -> RGBA 并转换为[0,1]浮点
                    row_dst[dst_idx + 0] = row_src[src_idx + 2] / 255.0f; // R
                    row_dst[dst_idx + 1] = row_src[src_idx + 1] / 255.0f; // G
                    row_dst[dst_idx + 2] = row_src[src_idx + 0] / 255.0f; // B
                    row_dst[dst_idx + 3] = row_src[src_idx + 3] / 255.0f; // A
                } else {
                    // RGBA 直接转换为[0,1]浮点
                    row_dst[dst_idx + 0] = row_src[src_idx + 0] / 255.0f; // R
                    row_dst[dst_idx + 1] = row_src[src_idx + 1] / 255.0f; // G
                    row_dst[dst_idx + 2] = row_src[src_idx + 2] / 255.0f; // B
                    row_dst[dst_idx + 3] = row_src[src_idx + 3] / 255.0f; // A
                }
            }
        }
    } else if (desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT) {
        // 浮点格式：每个像素16字节，直接复制
        size_t expected_row_bytes = desc.Width * 4 * sizeof(float);
        if (mapped.RowPitch < expected_row_bytes) {
            std::cerr << "Invalid RowPitch: " << mapped.RowPitch 
                      << " < expected: " << expected_row_bytes << std::endl;
            context->Unmap(staging_texture.Get(), 0);
            return std::vector<float>();
        }
        
        for (size_t y = 0; y < desc.Height; ++y) {
            const float* row_src = reinterpret_cast<const float*>(
                reinterpret_cast<const uint8_t*>(mapped.pData) + y * mapped.RowPitch);
            float* row_dst = pixels.data() + y * desc.Width * 4;
            memcpy(row_dst, row_src, expected_row_bytes);
        }
    } else {
        std::cerr << "Unsupported texture format: " << desc.Format << std::endl;
        context->Unmap(staging_texture.Get(), 0);
        return std::vector<float>();
    }
    
    context->Unmap(staging_texture.Get(), 0);
    
    return pixels;
}

// 辅助函数：从CUDA buffer读取像素数据到CPU
std::vector<float> readCudaBufferPixels(void* cuda_buffer, size_t buffer_size) {
    if (cuda_buffer == nullptr || buffer_size == 0) {
        std::cerr << "Invalid CUDA buffer parameters" << std::endl;
        return std::vector<float>();
    }
    
    // 计算像素数量 (buffer_size / sizeof(float))
    size_t pixel_count = buffer_size / sizeof(float);
    std::vector<float> cpu_pixels(pixel_count);
    
    // 从CUDA设备内存复制到CPU内存
    cudaError_t err = cudaMemcpy(cpu_pixels.data(), cuda_buffer, buffer_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return std::vector<float>();
    }
    
    return cpu_pixels;
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
        
        // 读取CUDA buffer像素数据到CPU
        auto cuda_pixels = readCudaBufferPixels(frame.cuda_buffer, frame.buffer_size);
        REQUIRE(!cuda_pixels.empty());
        
        uint32_t width = decoder.getWidth();
        uint32_t height = decoder.getHeight();
        size_t expected_pixel_count = width * height * 4; // RGBA
        REQUIRE(cuda_pixels.size() == expected_pixel_count);
        
        // 验证像素值范围[0,1]
        float min_val = 1.0f, max_val = 0.0f;
        size_t out_of_range_count = 0;
        
        for (size_t i = 0; i < cuda_pixels.size(); ++i) {
            float val = cuda_pixels[i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            
            if (val < 0.0f || val > 1.0f) {
                out_of_range_count++;
            }
        }
        
        std::cout << "CUDA buffer pixel analysis:" << std::endl;
        std::cout << "  Size: " << frame.buffer_size << " bytes (" << cuda_pixels.size() << " floats)" << std::endl;
        std::cout << "  Dimensions: " << width << "x" << height << " (RGBA)" << std::endl;
        std::cout << "  Value range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "  Out of range pixels: " << out_of_range_count << "/" << cuda_pixels.size() << std::endl;
        
        // 要求所有像素值都在[0,1]范围内
        REQUIRE(out_of_range_count == 0);
        REQUIRE(min_val >= 0.0f);
        REQUIRE(max_val <= 1.0f);
        
        // 读取输入RGB纹理进行对比验证
        auto rgb_pixels = readD3D11FloatTexturePixels(device, context.Get(), frame.rgb_frame.rgb_texture.Get());
        REQUIRE(!rgb_pixels.empty());
        REQUIRE(rgb_pixels.size() == cuda_pixels.size());
        
        // 对比RGB纹理和CUDA buffer的像素一致性
        // 注意：CUDA buffer使用BCHW布局，需要重新排列数据进行比较
        size_t diff_count = 0;
        float max_diff = 0.0f;
        const float tolerance = 1e-4f; // 浮点比较容差，考虑uint8->float转换精度
        
        uint32_t channel_size = width * height;
        
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t pixel_idx = y * width + x;
                
                // RGB纹理像素索引（RGBA交错）
                size_t rgb_idx = pixel_idx * 4;
                
                // CUDA buffer索引（BCHW布局）
                size_t cuda_r_idx = 0 * channel_size + pixel_idx;
                size_t cuda_g_idx = 1 * channel_size + pixel_idx;
                size_t cuda_b_idx = 2 * channel_size + pixel_idx;
                size_t cuda_a_idx = 3 * channel_size + pixel_idx;
                
                // 比较各通道
                float rgb_r = rgb_pixels[rgb_idx + 0];
                float rgb_g = rgb_pixels[rgb_idx + 1];
                float rgb_b = rgb_pixels[rgb_idx + 2];
                float rgb_a = rgb_pixels[rgb_idx + 3];
                
                float cuda_r = cuda_pixels[cuda_r_idx];
                float cuda_g = cuda_pixels[cuda_g_idx];
                float cuda_b = cuda_pixels[cuda_b_idx];
                float cuda_a = cuda_pixels[cuda_a_idx];
                
                // 计算各通道差异
                float diff_r = std::abs(rgb_r - cuda_r);
                float diff_g = std::abs(rgb_g - cuda_g);
                float diff_b = std::abs(rgb_b - cuda_b);
                float diff_a = std::abs(rgb_a - cuda_a);
                
                float pixel_max_diff = std::max({diff_r, diff_g, diff_b, diff_a});
                max_diff = std::max(max_diff, pixel_max_diff);
                
                if (pixel_max_diff > tolerance) {
                    diff_count++;
                }
            }
        }
        
        // 输出一些像素样本进行调试
        std::cout << "First few pixel samples (RGB vs CUDA BCHW):" << std::endl;
        for (int i = 0; i < 5 && i < width * height; ++i) {
            size_t rgb_idx = i * 4;
            size_t cuda_r_idx = 0 * channel_size + i;
            size_t cuda_g_idx = 1 * channel_size + i;
            size_t cuda_b_idx = 2 * channel_size + i;
            size_t cuda_a_idx = 3 * channel_size + i;
            
            std::cout << "Pixel " << i << ": RGB=(" 
                      << rgb_pixels[rgb_idx + 0] << ", " << rgb_pixels[rgb_idx + 1] << ", " 
                      << rgb_pixels[rgb_idx + 2] << ", " << rgb_pixels[rgb_idx + 3] << ") "
                      << "CUDA=(" << cuda_pixels[cuda_r_idx] << ", " << cuda_pixels[cuda_g_idx] << ", "
                      << cuda_pixels[cuda_b_idx] << ", " << cuda_pixels[cuda_a_idx] << ")" << std::endl;
        }
        
        std::cout << "RGB vs CUDA pixel comparison (BCHW layout):" << std::endl;
        std::cout << "  Max difference: " << max_diff << std::endl;
        std::cout << "  Different pixels (tolerance=" << tolerance << "): " << diff_count << "/" << (width * height) << std::endl;
        
        // 验证RGB纹理和CUDA buffer像素一致性
        // 考虑uint8->float转换精度，允许适当的差异
        double diff_ratio = static_cast<double>(diff_count) / (width * height);
        REQUIRE(diff_ratio < 0.05); // 允许不超过5%的像素有微小差异
        REQUIRE(max_diff < 0.01f);   // 最大差异不超过0.01（约2.5/255）
        
        // 释放AVFrame内存
        av_frame_free(&frame.rgb_frame.hw_frame.frame);
    }
    
    SECTION("Verify multiple frames conversion stability") {
        REQUIRE(decoder.open(TEST_FILE));
        
        const int test_frame_count = 5;
        std::vector<std::pair<float, float>> frame_ranges; // min, max for each frame
        
        for (int i = 0; i < test_frame_count; ++i) {
            FloatRgbVideoDecoder::DecodedFloatRgbFrame frame;
            bool success = decoder.readNextFrame(frame);
            if (!success) {
                std::cout << "End of video reached at frame " << i << std::endl;
                break;
            }
            
            REQUIRE(frame.is_valid);
            REQUIRE(frame.cuda_buffer != nullptr);
            
            // 读取CUDA buffer像素数据
            auto cuda_pixels = readCudaBufferPixels(frame.cuda_buffer, frame.buffer_size);
            REQUIRE(!cuda_pixels.empty());
            
            // 验证像素值范围
            float min_val = 1.0f, max_val = 0.0f;
            size_t out_of_range_count = 0;
            
            for (float val : cuda_pixels) {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                if (val < 0.0f || val > 1.0f) {
                    out_of_range_count++;
                }
            }
            
            frame_ranges.push_back({min_val, max_val});
            
            std::cout << "Frame " << i << " - Range: [" << min_val << ", " << max_val 
                      << "], Out of range: " << out_of_range_count << std::endl;
            
            // 每帧都必须满足范围要求
            REQUIRE(out_of_range_count == 0);
            REQUIRE(min_val >= 0.0f);
            REQUIRE(max_val <= 1.0f);
            
            // 释放AVFrame内存
            av_frame_free(&frame.rgb_frame.hw_frame.frame);
        }
        
        // 验证至少处理了3帧
        REQUIRE(frame_ranges.size() >= 3);
        
        std::cout << "Multi-frame test completed: " << frame_ranges.size() << " frames processed" << std::endl;
    }
}