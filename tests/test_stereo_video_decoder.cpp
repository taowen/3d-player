#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "../src/stereo_video_decoder.h"
#include <iostream>
#include <chrono>
#include <algorithm>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/rational.h>
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