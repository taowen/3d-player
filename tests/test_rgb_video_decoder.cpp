#include <catch2/catch_test_macros.hpp>
#include "../src/rgb_video_decoder.h"
#include <fstream>
#include <iostream>
#include <string>

const std::string TEST_FILE = "test_data/sample_hw.mkv";

TEST_CASE("RgbVideoDecoder basic functionality", "[rgb_video_decoder]") {
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
}

TEST_CASE("RgbVideoDecoder hardware decoder access", "[rgb_video_decoder]") {
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

TEST_CASE("RgbVideoDecoder error handling", "[rgb_video_decoder]") {
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