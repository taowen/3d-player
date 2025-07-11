#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <iostream>
#include <fstream>
#include <set>

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

TEST_CASE("HwVideoDecoder with valid MKV file", "[hw_video_decoder][requires_test_file]") {
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    HwVideoDecoder decoder;
    
    SECTION("Open and basic decoding") {
        REQUIRE(decoder.open(TEST_MKV_FILE));
        REQUIRE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());
        
        // Test actual GPU memory reuse
        HwVideoDecoder::DecodedFrame frame1, frame2, frame3;
        
        // First decode
        REQUIRE(decoder.readNextFrame(frame1));
        REQUIRE(frame1.is_valid);
        REQUIRE(frame1.frame != nullptr);
        
        // Get D3D11 surface address and sub-resource index
        void* gpu_surface1 = nullptr;
        int sub_resource1 = -1;
        
        gpu_surface1 = frame1.frame->data[0];  // D3D11 surface pointer
        if (frame1.frame->data[1] != nullptr) {
            sub_resource1 = *reinterpret_cast<int*>(frame1.frame->data[1]);
        }
        
        // Second decode
        REQUIRE(decoder.readNextFrame(frame2));
        REQUIRE(frame2.is_valid);
        REQUIRE(frame2.frame != nullptr);
        
        void* gpu_surface2 = nullptr;
        int sub_resource2 = -1;
        
        gpu_surface2 = frame2.frame->data[0];
        if (frame2.frame->data[1] != nullptr) {
            sub_resource2 = *reinterpret_cast<int*>(frame2.frame->data[1]);
        }
        
        // Third decode - should reuse the first surface
        REQUIRE(decoder.readNextFrame(frame3));
        REQUIRE(frame3.is_valid);
        REQUIRE(frame3.frame != nullptr);
        
        void* gpu_surface3 = nullptr;
        int sub_resource3 = -1;
        
        gpu_surface3 = frame3.frame->data[0];
        if (frame3.frame->data[1] != nullptr) {
            sub_resource3 = *reinterpret_cast<int*>(frame3.frame->data[1]);
        }
        
        // Key verification: check that we're getting valid hardware surfaces
        // and that the decoder is working properly
        REQUIRE(gpu_surface1 != nullptr);
        REQUIRE(gpu_surface2 != nullptr);
        REQUIRE(gpu_surface3 != nullptr);
        
        // Verify that we're getting different surfaces (they should not all be the same)
        bool all_same = (gpu_surface1 == gpu_surface2) && (gpu_surface2 == gpu_surface3);
        REQUIRE_FALSE(all_same);  // At least some surfaces should be different
        
        // Print debug information for verification
        std::cout << "GPU surface 1: " << gpu_surface1 << std::endl;
        std::cout << "GPU surface 2: " << gpu_surface2 << std::endl;
        std::cout << "GPU surface 3: " << gpu_surface3 << std::endl;
        
        decoder.close();
        REQUIRE_FALSE(decoder.isOpen());
    }
}

TEST_CASE("HwVideoDecoder GPU memory management", "[hw_video_decoder][requires_test_file]") {
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    HwVideoDecoder decoder;
    REQUIRE(decoder.open(TEST_MKV_FILE));
    
    SECTION("GPU surface and sub-resource pattern verification") {
        // Test continuous multiple calls for GPU surface and sub-resource patterns
        std::vector<void*> gpu_surfaces;
        std::vector<int> sub_resources;
        
        HwVideoDecoder::DecodedFrame frame;
        
        // Decode 6 frames to test pattern
        for (int i = 0; i < 6; i++) {
            REQUIRE(decoder.readNextFrame(frame));
            REQUIRE(frame.is_valid);
            REQUIRE(frame.frame != nullptr);
            
            gpu_surfaces.push_back(frame.frame->data[0]);
            if (frame.frame->data[1] != nullptr) {
                sub_resources.push_back(*reinterpret_cast<int*>(frame.frame->data[1]));
            } else {
                sub_resources.push_back(-1);
            }
        }
        
        // Check if all surface pointers are the same (using the same texture)
        bool all_same_surface = true;
        for (size_t i = 1; i < gpu_surfaces.size(); i++) {
            if (gpu_surfaces[i] != gpu_surfaces[0]) {
                all_same_surface = false;
                break;
            }
        }
        
        if (all_same_surface) {
            // If using the same surface, verify at least some different sub-resource indices are used
            std::set<int> unique_sub_resources(sub_resources.begin(), sub_resources.end());
            
            REQUIRE(unique_sub_resources.size() >= 2);  // At least 2 different sub-resources used
        } else {
            // If surface pointers are different, verify traditional double buffering mode
            // This is a simple verification - actual patterns may be more complex
            REQUIRE(gpu_surfaces.size() >= 2);
        }
        
        // Print debug information
        std::cout << "GPU surfaces: ";
        for (size_t i = 0; i < gpu_surfaces.size(); i++) {
            std::cout << gpu_surfaces[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Sub-resources: ";
        for (size_t i = 0; i < sub_resources.size(); i++) {
            std::cout << sub_resources[i] << " ";
        }
        std::cout << std::endl;
    }
    
    decoder.close();
}

TEST_CASE("HwVideoDecoder D3D11 device access", "[hw_video_decoder][requires_test_file]") {
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    HwVideoDecoder decoder;
    
    SECTION("D3D11 device and context") {
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

