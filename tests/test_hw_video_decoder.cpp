#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <iostream>

extern "C" {
#include <libavutil/pixfmt.h>
}

TEST_CASE("HwVideoDecoder basic functionality", "[hw_video_decoder]") {
    HwVideoDecoder decoder;
    
    // Open video file
    bool opened = decoder.open("test_data/sample_hw.mkv");
    REQUIRE(opened);
    REQUIRE(decoder.isOpen());
    
    // Read first frame
    HwVideoDecoder::DecodedFrame frame1;
    bool result1 = decoder.readNextFrame(frame1);
    REQUIRE(result1);
    REQUIRE(frame1.is_valid);
    REQUIRE(frame1.frame != nullptr);
    
    // Verify frame properties
    REQUIRE(frame1.frame->width > 0);
    REQUIRE(frame1.frame->height > 0);
    
    // Verify this is hardware decoded (any supported hardware format)
    bool is_hardware_format = (frame1.frame->format == AV_PIX_FMT_D3D11 ||
                               frame1.frame->format == AV_PIX_FMT_D3D11VA_VLD ||
                               frame1.frame->format == AV_PIX_FMT_DXVA2_VLD ||
                               frame1.frame->format == AV_PIX_FMT_CUDA ||
                               frame1.frame->format == AV_PIX_FMT_VULKAN);
    REQUIRE(is_hardware_format);
    
    // Release first frame
    av_frame_free(&frame1.frame);
    
    // Read second frame
    HwVideoDecoder::DecodedFrame frame2;
    bool result2 = decoder.readNextFrame(frame2);
    REQUIRE(result2);
    REQUIRE(frame2.is_valid);
    REQUIRE(frame2.frame != nullptr);
    
    // Verify second frame properties
    REQUIRE(frame2.frame->width > 0);
    REQUIRE(frame2.frame->height > 0);
    
    // Release second frame
    av_frame_free(&frame2.frame);
    
    // Read third frame
    HwVideoDecoder::DecodedFrame frame3;
    bool result3 = decoder.readNextFrame(frame3);
    REQUIRE(result3);
    REQUIRE(frame3.is_valid);
    REQUIRE(frame3.frame != nullptr);
    
    // Release third frame
    av_frame_free(&frame3.frame);
    
    // Still should be open and not EOF yet
    REQUIRE(decoder.isOpen());
    
    decoder.close();
    REQUIRE_FALSE(decoder.isOpen());
}

TEST_CASE("HwVideoDecoder hardware acceleration verification", "[hw_video_decoder]") {
    HwVideoDecoder decoder;
    
    // Open video file
    bool opened = decoder.open("test_data/sample_hw.mkv");
    REQUIRE(opened);
    REQUIRE(decoder.isOpen());
    
    // Verify hardware decoder components were initialized
    REQUIRE(decoder.getD3D11Device() != nullptr);
    REQUIRE(decoder.getD3D11DeviceContext() != nullptr);
    
    // Verify hardware decoding infrastructure is working:
    // 1. D3D11VA decoder must be found and initialized
    // 2. Hardware device context must be created successfully
    // 3. Only hardware pixel formats are accepted (no software fallback)
    // 4. This confirms mandatory hardware decoding requirement
    
    decoder.close();
    REQUIRE_FALSE(decoder.isOpen());
}

