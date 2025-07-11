#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <fstream>

const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

TEST_CASE("HwVideoDecoder basic functionality", "[hw_video_decoder]") {
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    HwVideoDecoder decoder;
    
    SECTION("Complete video decoding test") {
        // Initial state
        REQUIRE_FALSE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());
        
        // Open file
        REQUIRE(decoder.open(TEST_MKV_FILE));
        REQUIRE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());
        
        // Read first frame
        HwVideoDecoder::DecodedFrame frame;
        REQUIRE(decoder.readNextFrame(frame));
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        REQUIRE(frame.frame->width > 0);
        REQUIRE(frame.frame->height > 0);
        REQUIRE(frame.frame->data[0] != nullptr);
        
        // Close
        decoder.close();
        REQUIRE_FALSE(decoder.isOpen());
    }
}

