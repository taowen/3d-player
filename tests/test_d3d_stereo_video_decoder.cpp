#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#include "../src/d3d_stereo_video_decoder.h"

// D3D11 headers
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

TEST_CASE("D3dStereoVideoDecoder Basic Functionality", "[d3d_stereo_video_decoder][test_d3d_stereo_video_decoder.cpp]") {
    
    SECTION("Constructor and Destructor") {
        D3dStereoVideoDecoder decoder;
        REQUIRE_FALSE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());
    }
    
    SECTION("Open and Close Video File") {
        D3dStereoVideoDecoder decoder;
        
        // 测试打开不存在的文件
        REQUIRE_FALSE(decoder.open("nonexistent_file.mkv"));
        REQUIRE_FALSE(decoder.isOpen());
        
        // 测试打开存在的测试文件
        bool opened = decoder.open("test_data/sample_hw.mkv");
        if (opened) {
            REQUIRE(decoder.isOpen());
            REQUIRE_FALSE(decoder.isEOF());
            
            // 检查视频尺寸
            int width = decoder.getWidth();
            int height = decoder.getHeight();
            REQUIRE(width > 0);
            REQUIRE(height > 0);
            std::cout << "Video dimensions: " << width << "x" << height << std::endl;
            
            // 检查 D3D11 设备
            ID3D11Device* device = decoder.getD3D11Device();
            REQUIRE(device != nullptr);
            
            // 检查时间基准
            AVRational timebase = decoder.getVideoTimeBase();
            REQUIRE(timebase.den > 0);
            std::cout << "Video timebase: " << timebase.num << "/" << timebase.den << std::endl;
            
            decoder.close();
            REQUIRE_FALSE(decoder.isOpen());
        } else {
            std::cout << "Warning: Could not open test video file, skipping open/close tests" << std::endl;
        }
    }
}

TEST_CASE("D3dStereoVideoDecoder Frame Reading", "[d3d_stereo_video_decoder][test_d3d_stereo_video_decoder.cpp]") {
    
    SECTION("Read Single Frame with D3D Conversion") {
        D3dStereoVideoDecoder decoder;
        
        bool opened = decoder.open("test_data/sample_hw.mkv");
        if (!opened) {
            std::cout << "Warning: Could not open test video file, skipping frame reading tests" << std::endl;
            return;
        }
        
        REQUIRE(decoder.isOpen());
        
        DecodedStereoFrameD3D frame;
        bool frame_read = decoder.readNextFrame(frame);
        
        REQUIRE(frame_read);
        REQUIRE(frame.stereo_frame.is_valid);
        REQUIRE(frame.stereo_frame.input_frame != nullptr);
        REQUIRE(frame.stereo_frame.cuda_output_buffer != nullptr);
        
        // 检查 D3D11 转换结果
        if (frame.d3d_conversion_valid) {
            REQUIRE(frame.d3d_texture != nullptr);
            
            // 验证纹理属性
            D3D11_TEXTURE2D_DESC texture_desc;
            frame.d3d_texture->GetDesc(&texture_desc);
            
            REQUIRE(texture_desc.Width == static_cast<UINT>(decoder.getWidth()));
            REQUIRE(texture_desc.Height == static_cast<UINT>(decoder.getHeight()));
            REQUIRE(texture_desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT);
            
            std::cout << "D3D11 texture created successfully: " 
                      << texture_desc.Width << "x" << texture_desc.Height 
                      << " format=" << texture_desc.Format << std::endl;
        } else {
            std::cout << "Warning: D3D11 conversion failed, but stereo frame is valid" << std::endl;
        }
        
        decoder.close();
    }
    
    SECTION("Read Multiple Frames") {
        D3dStereoVideoDecoder decoder;
        
        bool opened = decoder.open("test_data/sample_hw.mkv");
        if (!opened) {
            std::cout << "Warning: Could not open test video file, skipping multiple frame tests" << std::endl;
            return;
        }
        
        REQUIRE(decoder.isOpen());
        
        int frame_count = 0;
        int successful_d3d_conversions = 0;
        DecodedStereoFrameD3D frame;
        
        // 读取前5帧进行测试
        while (frame_count < 5 && decoder.readNextFrame(frame)) {
            frame_count++;
            
            REQUIRE(frame.stereo_frame.is_valid);
            REQUIRE(frame.stereo_frame.input_frame != nullptr);
            REQUIRE(frame.stereo_frame.cuda_output_buffer != nullptr);
            
            if (frame.d3d_conversion_valid) {
                successful_d3d_conversions++;
                REQUIRE(frame.d3d_texture != nullptr);
            }
            
            std::cout << "Frame " << frame_count << ": stereo_valid=" 
                      << frame.stereo_frame.is_valid << ", d3d_valid=" 
                      << frame.d3d_conversion_valid << std::endl;
        }
        
        REQUIRE(frame_count > 0);
        std::cout << "Successfully read " << frame_count << " frames, " 
                  << successful_d3d_conversions << " D3D conversions succeeded" << std::endl;
        
        decoder.close();
    }
}

TEST_CASE("D3dStereoVideoDecoder Error Handling", "[d3d_stereo_video_decoder][test_d3d_stereo_video_decoder.cpp]") {
    
    SECTION("Read Frame Without Opening") {
        D3dStereoVideoDecoder decoder;
        
        DecodedStereoFrameD3D frame;
        REQUIRE_FALSE(decoder.readNextFrame(frame));
        REQUIRE_FALSE(frame.stereo_frame.is_valid);
        REQUIRE_FALSE(frame.d3d_conversion_valid);
        REQUIRE(frame.d3d_texture == nullptr);
    }
    
    SECTION("Multiple Close Calls") {
        D3dStereoVideoDecoder decoder;
        
        // 多次调用 close 应该是安全的
        decoder.close();
        decoder.close();
        decoder.close();
        
        REQUIRE_FALSE(decoder.isOpen());
    }
    
    SECTION("Properties Without Opening") {
        D3dStereoVideoDecoder decoder;
        
        REQUIRE(decoder.getWidth() == 0);
        REQUIRE(decoder.getHeight() == 0);
        REQUIRE(decoder.getD3D11Device() == nullptr);
        
        AVRational timebase = decoder.getVideoTimeBase();
        REQUIRE(timebase.num == 0);
        REQUIRE(timebase.den == 1);
    }
}

TEST_CASE("D3dStereoVideoDecoder Performance", "[d3d_stereo_video_decoder][performance][test_d3d_stereo_video_decoder.cpp]") {
    
    SECTION("Frame Reading Performance") {
        D3dStereoVideoDecoder decoder;
        
        bool opened = decoder.open("test_data/sample_hw.mkv");
        if (!opened) {
            std::cout << "Warning: Could not open test video file, skipping performance tests" << std::endl;
            return;
        }
        
        const int test_frame_count = 10;
        DecodedStereoFrameD3D frame;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int successful_reads = 0;
        for (int i = 0; i < test_frame_count; ++i) {
            if (decoder.readNextFrame(frame)) {
                successful_reads++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        REQUIRE(successful_reads > 0);
        
        double avg_time_per_frame = static_cast<double>(duration.count()) / successful_reads;
        std::cout << "Performance: " << successful_reads << " frames in " 
                  << duration.count() << "ms, avg " << avg_time_per_frame << "ms per frame" << std::endl;
        
        // 基本性能要求：平均每帧处理时间不超过100ms
        REQUIRE(avg_time_per_frame < 100.0);
        
        decoder.close();
    }
}