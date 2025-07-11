#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <fstream>
#include <algorithm>
#include <iostream>

extern "C" {
#include <libavutil/pixfmt.h>
}

const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

TEST_CASE("HwVideoDecoder basic functionality", "[hw_video_decoder]") {
    HwVideoDecoder decoder;
    
    SECTION("Initial state") {
        REQUIRE_FALSE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());
        REQUIRE(decoder.getStreamReader() != nullptr);
    }
    
    SECTION("Open non-existent file") {
        REQUIRE_FALSE(decoder.open("non_existent_file.mkv"));
        REQUIRE_FALSE(decoder.isOpen());
    }
}

TEST_CASE("HwVideoDecoder DirectX 11 hardware decoding (MANDATORY - NO FALLBACKS)", "[hw_video_decoder][requires_test_file]") {
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    HwVideoDecoder decoder;
    
    SECTION("MANDATORY: DirectX 11 hardware decoding must work or fail completely") {
        REQUIRE(decoder.open(TEST_MKV_FILE));
        REQUIRE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());
        
        // Must have D3D11 device and context available
        auto device = decoder.getD3D11Device();
        auto context = decoder.getD3D11Context();
        REQUIRE(device != nullptr);
        REQUIRE(context != nullptr);
        
        // Test decoding and verify hardware format
        HwVideoDecoder::DecodedFrame frame;
        REQUIRE(decoder.readNextFrame(frame));
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        
        // Print actual pixel format for debugging
        std::cout << "Actual pixel format: " << frame.frame->format << std::endl;
        std::cout << "AV_PIX_FMT_D3D11: " << AV_PIX_FMT_D3D11 << std::endl;
        std::cout << "AV_PIX_FMT_DXVA2_VLD: " << AV_PIX_FMT_DXVA2_VLD << std::endl;
        std::cout << "AV_PIX_FMT_QSV: " << AV_PIX_FMT_QSV << std::endl;
        std::cout << "AV_PIX_FMT_YUV420P: " << AV_PIX_FMT_YUV420P << std::endl;
        std::cout << "AV_PIX_FMT_NV12: " << AV_PIX_FMT_NV12 << std::endl;
        
        // Verify frame properties
        REQUIRE(frame.frame->width > 0);
        REQUIRE(frame.frame->height > 0);
        REQUIRE(frame.frame->format != AV_PIX_FMT_NONE);
        
        // CRITICAL: MUST be using DirectX 11 hardware format - NO EXCEPTIONS
        bool is_d3d11_hardware = (frame.frame->format == AV_PIX_FMT_D3D11 || 
                                 frame.frame->format == AV_PIX_FMT_D3D11VA_VLD);
        
        if (!is_d3d11_hardware) {
            std::cout << "❌ FATAL: Expected DirectX 11 hardware format but got: " << frame.frame->format << std::endl;
            std::cout << "❌ DirectX 11 hardware acceleration is REQUIRED!" << std::endl;
        }
        REQUIRE(is_d3d11_hardware);
        
        if (frame.frame->format == AV_PIX_FMT_D3D11) {
            std::cout << "✅ SUCCESS: Using AV_PIX_FMT_D3D11 DirectX 11 hardware decoding!" << std::endl;
        } else if (frame.frame->format == AV_PIX_FMT_D3D11VA_VLD) {
            std::cout << "✅ SUCCESS: Using AV_PIX_FMT_D3D11VA_VLD DirectX 11 hardware decoding!" << std::endl;
        }
        
        // Verify GPU surface is allocated
        REQUIRE(frame.frame->data[0] != nullptr);
        
        // Verify frame has reasonable dimensions
        REQUIRE(frame.frame->width >= 64);
        REQUIRE(frame.frame->height >= 64);
        REQUIRE(frame.frame->width <= 4096);
        REQUIRE(frame.frame->height <= 4096);
        
        // Test multiple frames to ensure consistent D3D11 hardware decoding
        for (int i = 1; i < 3; i++) {
            HwVideoDecoder::DecodedFrame next_frame;
            REQUIRE(decoder.readNextFrame(next_frame));
            REQUIRE(next_frame.is_valid);
            REQUIRE(next_frame.frame != nullptr);
            REQUIRE(next_frame.frame->data[0] != nullptr);
            // Must consistently use D3D11 hardware format
            bool is_d3d11_hardware = (next_frame.frame->format == AV_PIX_FMT_D3D11 || 
                                     next_frame.frame->format == AV_PIX_FMT_D3D11VA_VLD);
            REQUIRE(is_d3d11_hardware);
            // Format should be consistent across frames
            REQUIRE(next_frame.frame->format == frame.frame->format);
        }
        
        decoder.close();
        REQUIRE_FALSE(decoder.isOpen());
    }
}

TEST_CASE("HwVideoDecoder D3D11 device access", "[hw_video_decoder][requires_test_file]") {
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    HwVideoDecoder decoder;
    
    SECTION("D3D11 device and context availability") {
        REQUIRE(decoder.open(TEST_MKV_FILE));
        
        // Should be able to get D3D11 device and context
        auto device = decoder.getD3D11Device();
        auto context = decoder.getD3D11Context();
        
        REQUIRE(device != nullptr);
        REQUIRE(context != nullptr);
        
        // Test basic device functionality
        UINT creation_flags = device->GetCreationFlags();
        D3D_FEATURE_LEVEL level = device->GetFeatureLevel();
        
        std::cout << "D3D11 device creation flags: " << creation_flags << std::endl;
        std::cout << "D3D11 feature level: " << level << std::endl;
        
        // Verify feature level is reasonable
        REQUIRE(level >= D3D_FEATURE_LEVEL_10_0);
        
        decoder.close();
    }
}

TEST_CASE("HwVideoDecoder error handling", "[hw_video_decoder]") {
    HwVideoDecoder decoder;
    
    SECTION("Operations on closed decoder") {
        REQUIRE_FALSE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());
        
        HwVideoDecoder::DecodedFrame frame;
        REQUIRE_FALSE(decoder.readNextFrame(frame));
        REQUIRE_FALSE(frame.is_valid);
        
        // Should return nullptr for D3D11 resources when closed
        REQUIRE(decoder.getD3D11Device() == nullptr);
        REQUIRE(decoder.getD3D11Context() == nullptr);
        
        // These should not crash
        decoder.close();
    }
}

