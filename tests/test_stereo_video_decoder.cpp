#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "../src/stereo_video_decoder.h"
#include <iostream>
#include <chrono>

TEST_CASE("Stereo Video Decoder Frame Reading", "[stereo_video_decoder][frames]") {
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
            REQUIRE(frame.pts_seconds >= 0.0);
            
            D3D11_TEXTURE2D_DESC texture_desc;
            frame.stereo_texture->GetDesc(&texture_desc);
            
            REQUIRE(texture_desc.Width == decoder.getWidth());
            REQUIRE(texture_desc.Height == decoder.getHeight());
            REQUIRE(texture_desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT);
            
            std::cout << "Frame " << frame_count 
                      << ": " << texture_desc.Width << "x" << texture_desc.Height
                      << ", PTS: " << frame.pts_seconds << "s"
                      << ", Processing time: " << duration.count() << "ms" << std::endl;
            
            frame_count++;
        }
        
        REQUIRE(frame_count > 0);
        std::cout << "Successfully processed " << frame_count << " frames" << std::endl;
        
        decoder.close();
    }
}