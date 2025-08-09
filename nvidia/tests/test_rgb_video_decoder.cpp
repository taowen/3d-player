#include <catch2/catch_test_macros.hpp>
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

// 辅助函数：从 D3D11 纹理读取像素数据
std::vector<uint8_t> readD3D11TexturePixels(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture) {
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
        return std::vector<uint8_t>();
    }
    
    // 复制纹理到 staging
    context->CopyResource(staging_texture.Get(), texture);
    
    // 映射并读取数据
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture, HRESULT: 0x" << std::hex << hr << std::endl;
        return std::vector<uint8_t>();
    }
    
    std::vector<uint8_t> pixels;
    pixels.resize(desc.Height * mapped.RowPitch);
    memcpy(pixels.data(), mapped.pData, pixels.size());
    
    context->Unmap(staging_texture.Get(), 0);
    
    return pixels;
}

// 辅助函数：从 D3D11 硬件帧读取 YUV 数据
std::vector<uint8_t> readD3D11YuvData(AVFrame* hw_frame) {
    if (hw_frame->format != AV_PIX_FMT_D3D11) {
        return std::vector<uint8_t>();
    }
    
    // 获取 D3D11 纹理
    ID3D11Texture2D* texture = (ID3D11Texture2D*)hw_frame->data[0];
    int array_index = (int)(intptr_t)hw_frame->data[1];
    
    // 获取 D3D11 设备和上下文
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frame->hw_frames_ctx->data;
    AVD3D11VADeviceContext* d3d11_device_ctx = (AVD3D11VADeviceContext*)frames_ctx->device_ctx->hwctx;
    ID3D11Device* device = d3d11_device_ctx->device;
    ID3D11DeviceContext* context = d3d11_device_ctx->device_context;
    
    // 获取纹理描述
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    // 创建 staging 纹理
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    staging_desc.ArraySize = 1;
    
    ComPtr<ID3D11Texture2D> staging_texture;
    HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        return std::vector<uint8_t>();
    }
    
    // 复制子资源
    context->CopySubresourceRegion(
        staging_texture.Get(), 0, 0, 0, 0,
        texture, array_index,
        nullptr
    );
    
    // 映射并读取数据
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        return std::vector<uint8_t>();
    }
    
    // NV12 格式：Y 平面 + UV 交错平面
    int y_size = hw_frame->height * mapped.RowPitch;
    int uv_size = (hw_frame->height / 2) * mapped.RowPitch;
    std::vector<uint8_t> yuv_data(y_size + uv_size);
    memcpy(yuv_data.data(), mapped.pData, yuv_data.size());
    
    context->Unmap(staging_texture.Get(), 0);
    
    return yuv_data;
}

// 辅助函数：使用 FFmpeg swscale 进行 YUV 到 RGB 转换
std::vector<uint8_t> convertYuvToRgbWithFFmpeg(AVFrame* hw_frame, const std::vector<uint8_t>& yuv_data) {
    int width = hw_frame->width;
    int height = hw_frame->height;
    
    // 创建软件 YUV frame
    AVFrame* yuv_frame = av_frame_alloc();
    yuv_frame->format = AV_PIX_FMT_NV12;  // D3D11 默认格式
    yuv_frame->width = width;
    yuv_frame->height = height;
    av_frame_get_buffer(yuv_frame, 32);
    
    // 获取 RowPitch（从 readD3D11YuvData 函数知道）
    int row_pitch = yuv_data.size() / (height + height / 2);  // NV12 格式高度为 1.5 倍
    
    // 复制 Y 平面数据，考虑 padding
    for (int y = 0; y < height; ++y) {
        memcpy(yuv_frame->data[0] + y * yuv_frame->linesize[0],
               yuv_data.data() + y * row_pitch,
               width);
    }
    
    // 复制 UV 交错平面数据
    int uv_offset = height * row_pitch;
    for (int y = 0; y < height / 2; ++y) {
        memcpy(yuv_frame->data[1] + y * yuv_frame->linesize[1],
               yuv_data.data() + uv_offset + y * row_pitch,
               width);
    }
    
    // 创建 RGB frame
    AVFrame* rgb_frame = av_frame_alloc();
    rgb_frame->format = AV_PIX_FMT_BGRA;  // 对应 DXGI_FORMAT_B8G8R8A8_UNORM
    rgb_frame->width = width;
    rgb_frame->height = height;
    av_frame_get_buffer(rgb_frame, 32);
    
    // 创建 swscale 上下文
    SwsContext* sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_NV12,
        width, height, AV_PIX_FMT_BGRA,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    
    if (!sws_ctx) {
        av_frame_free(&yuv_frame);
        av_frame_free(&rgb_frame);
        return std::vector<uint8_t>();
    }
    
    // 执行转换
    sws_scale(sws_ctx, yuv_frame->data, yuv_frame->linesize, 0, height,
              rgb_frame->data, rgb_frame->linesize);
    
    // 复制数据到 vector
    std::vector<uint8_t> pixels;
    pixels.resize(height * rgb_frame->linesize[0]);
    memcpy(pixels.data(), rgb_frame->data[0], pixels.size());
    
    sws_freeContext(sws_ctx);
    av_frame_free(&yuv_frame);
    av_frame_free(&rgb_frame);
    
    return pixels;
}

// 辅助函数：比较两个像素数组
bool comparePixels(const std::vector<uint8_t>& pixels1, const std::vector<uint8_t>& pixels2,
                   int width, int height, int stride1, int stride2, double tolerance = 5.0) {
    int total_diff = 0;
    int max_diff = 0;
    int pixel_count = 0;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int offset1 = y * stride1 + x * 4;  // BGRA format
            int offset2 = y * stride2 + x * 4;  // BGRA format
            
            // 比较 BGR 通道（忽略 Alpha）
            for (int c = 0; c < 3; ++c) {
                int diff = std::abs(pixels1[offset1 + c] - pixels2[offset2 + c]);
                total_diff += diff;
                max_diff = (std::max)(max_diff, diff);
            }
            pixel_count++;
        }
    }
    
    double avg_diff = (double)total_diff / (pixel_count * 3);
    
    
    return avg_diff <= tolerance;
}

TEST_CASE("RgbVideoDecoder basic functionality", "[rgb_video_decoder][test_rgb_video_decoder.cpp]") {
    RgbVideoDecoder decoder;

    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(decoder.open("nonexistent_file.mkv"));
        REQUIRE_FALSE(decoder.isOpen());
    }

    SECTION("Open valid test file and decode frames") {
        std::ifstream file(TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << TEST_FILE << " not found");
        }
        REQUIRE(decoder.open(TEST_FILE));
        REQUIRE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());

        RgbVideoDecoder::DecodedRgbFrame frame;
        int frame_count = 0;
        int max_frames = 10;
        while (decoder.readNextFrame(frame) && frame_count < max_frames) {
            REQUIRE(frame.is_valid);
            REQUIRE(frame.hw_frame.frame != nullptr);
            REQUIRE(frame.rgb_texture != nullptr);
            av_frame_free(&frame.hw_frame.frame);
            frame_count++;
        }
        REQUIRE(frame_count > 0);
        decoder.close();
        REQUIRE_FALSE(decoder.isOpen());
    }
    
    SECTION("Verify RGB color conversion accuracy") {
        std::ifstream file(TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << TEST_FILE << " not found");
        }
        
        // 打开 RGB 解码器
        REQUIRE(decoder.open(TEST_FILE));
        REQUIRE(decoder.isOpen());
        
        // 创建独立的硬件解码器用于获取 YUV 帧（使用相同的设备）
        HwVideoDecoder hw_decoder;
        REQUIRE(hw_decoder.open(TEST_FILE));
        
        // 获取 D3D11 设备和上下文（使用RGB解码器的设备）
        ID3D11Device* device = decoder.getHwDecoder()->getD3D11Device();
        ID3D11DeviceContext* context = decoder.getHwDecoder()->getD3D11DeviceContext();
        REQUIRE(device != nullptr);
        REQUIRE(context != nullptr);
        
        // 解码并比较前几帧
        int frames_to_test = 1;
        for (int i = 0; i < frames_to_test; ++i) {
            // 从硬件解码器获取 YUV 帧
            HwVideoDecoder::DecodedFrame hw_frame;
            REQUIRE(hw_decoder.readNextFrame(hw_frame));
            
            // 从 RGB 解码器获取 RGB 帧
            RgbVideoDecoder::DecodedRgbFrame rgb_frame;
            REQUIRE(decoder.readNextFrame(rgb_frame));
            
            // 从硬件帧读取 YUV 数据
            std::vector<uint8_t> yuv_data = readD3D11YuvData(hw_frame.frame);
            REQUIRE(!yuv_data.empty());
            
            // 使用 FFmpeg 进行参考转换
            std::vector<uint8_t> ffmpeg_pixels = convertYuvToRgbWithFFmpeg(hw_frame.frame, yuv_data);
            REQUIRE(!ffmpeg_pixels.empty());
            
            // 从 D3D11 纹理读取 RGB 像素
            std::vector<uint8_t> d3d11_pixels = readD3D11TexturePixels(device, context, rgb_frame.rgb_texture.Get());
            REQUIRE(!d3d11_pixels.empty());
            
            // 比较结果
            int width = decoder.getWidth();
            int height = decoder.getHeight();
            int ffmpeg_stride = ffmpeg_pixels.size() / height;
            int d3d11_stride = d3d11_pixels.size() / height;
            
            // 比较结果，允许硬件和软件转换之间的色彩差异
            REQUIRE(comparePixels(ffmpeg_pixels, d3d11_pixels, width, height, ffmpeg_stride, d3d11_stride, 20.0));
            
            // 清理
            av_frame_free(&hw_frame.frame);
            av_frame_free(&rgb_frame.hw_frame.frame);
        }
        
        hw_decoder.close();
        decoder.close();
    }
}

TEST_CASE("RgbVideoDecoder hardware decoder access", "[rgb_video_decoder][test_rgb_video_decoder.cpp]") {
    RgbVideoDecoder decoder;
    HwVideoDecoder* hw_decoder = decoder.getHwDecoder();
    REQUIRE(hw_decoder != nullptr);

    std::ifstream file(TEST_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_FILE << " not found");
    }
    REQUIRE(decoder.open(TEST_FILE));
    REQUIRE(hw_decoder->isOpen());
    REQUIRE(hw_decoder->getStreamReader() != nullptr);
    decoder.close();
}

TEST_CASE("RgbVideoDecoder error handling", "[rgb_video_decoder][test_rgb_video_decoder.cpp]") {
    RgbVideoDecoder decoder;
    RgbVideoDecoder::DecodedRgbFrame frame;
    REQUIRE_FALSE(decoder.readNextFrame(frame));
    REQUIRE_FALSE(decoder.isOpen());
    REQUIRE_FALSE(decoder.isEOF());
    REQUIRE(decoder.getWidth() == 0);
    REQUIRE(decoder.getHeight() == 0);
    decoder.close();
    decoder.close(); // Should be safe
} 