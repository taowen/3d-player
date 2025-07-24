#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "../src/stereo_video_decoder.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <vector>
#include <wrl/client.h>
#include <cuda_runtime.h>

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



// 辅助函数：将CUDA浮点缓冲区保存为BMP文件
static void saveCudaFloatBufferToBMP(void* cuda_buffer, int width, int height, const std::string& filename) {
    if (!cuda_buffer) return;
    
    // 分配主机内存
    size_t total_pixels = width * height * 4; // RGBA
    std::vector<float> host_data(total_pixels);
    
    // 从GPU复制到CPU
    cudaError_t cuda_err = cudaMemcpy(host_data.data(), cuda_buffer, 
                                      total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy CUDA buffer to host: " << cudaGetErrorString(cuda_err) << std::endl;
        return;
    }
    
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
            for (int x = 0; x < width; ++x) {
                size_t pixel_idx = (y * width + x) * 4;
                
                // 浮点值[0,1]转换为8位整数[0,255]，RGBA -> BGR
                uint8_t r = static_cast<uint8_t>(std::round(host_data[pixel_idx + 0] * 255.0f));
                uint8_t g = static_cast<uint8_t>(std::round(host_data[pixel_idx + 1] * 255.0f));
                uint8_t b = static_cast<uint8_t>(std::round(host_data[pixel_idx + 2] * 255.0f));
                
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
            REQUIRE(frame.cuda_output_buffer != nullptr);
            REQUIRE(frame.cuda_output_size > 0);
            REQUIRE(frame.output_width > 0);
            REQUIRE(frame.output_height > 0);
            
            // 计算PTS时间戳
            double pts_seconds = 0.0;
            if (frame.frame && frame.frame->pts != AV_NOPTS_VALUE) {
                AVRational time_base = decoder.getVideoTimeBase();
                pts_seconds = frame.frame->pts * av_q2d(time_base);
            }
            REQUIRE(pts_seconds >= 0.0);
            
            // 验证 CUDA 缓冲区尺寸
            REQUIRE(frame.output_width == decoder.getWidth());
            REQUIRE(frame.output_height == decoder.getHeight());
            
            // 验证缓冲区大小（4个浮点数 RGBA * 宽 * 高）
            size_t expected_size = frame.output_width * frame.output_height * 4 * sizeof(float);
            REQUIRE(frame.cuda_output_size == expected_size);
            
            // 验证 CUDA 缓冲区像素数据
            size_t total_pixels = frame.output_width * frame.output_height * 4; // RGBA
            std::vector<float> host_data(total_pixels);
            
            // 从GPU复制到CPU进行验证
            cudaError_t cuda_err = cudaMemcpy(host_data.data(), frame.cuda_output_buffer, 
                                              total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
            REQUIRE(cuda_err == cudaSuccess);
            
            // 检查前几个像素值，确保不全为零且在合理范围内
            bool has_non_zero_pixel = false;
            bool has_valid_range_pixel = false;
            const int sample_pixels = (std::min)(100, static_cast<int>(frame.output_width * frame.output_height));
            
            for (int i = 0; i < sample_pixels; i++) {
                float r = host_data[i * 4 + 0];
                float g = host_data[i * 4 + 1]; 
                float b = host_data[i * 4 + 2];
                float a = host_data[i * 4 + 3];
                
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
            
            REQUIRE(has_non_zero_pixel);
            REQUIRE(has_valid_range_pixel);
            
            // 保存前3帧 CUDA 输出为BMP文件用于肉眼比对
            if (frame_count < 3) {
                std::string cuda_output_filename = "cuda_stereo_output_frame_" + std::to_string(frame_count) + ".bmp";
                saveCudaFloatBufferToBMP(frame.cuda_output_buffer, frame.output_width, frame.output_height, 
                                       cuda_output_filename);
                std::cout << "Saved CUDA stereo output frame: " << cuda_output_filename << std::endl;
            }
            
            std::cout << "Frame " << frame_count 
                      << ": " << frame.output_width << "x" << frame.output_height
                      << ", PTS: " << pts_seconds << "s"
                      << ", Processing time: " << duration.count() << "ms"
                      << ", CUDA buffer size: " << frame.cuda_output_size << " bytes"
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