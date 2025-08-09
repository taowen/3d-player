#include <catch2/catch_test_macros.hpp>
#include "../src/rgb_video_decoder.h"
#include <fstream>
const std::string TEST_FILE="test_data/sample_hw.mkv";
TEST_CASE("RgbVideoDecoder first frame (intel)", "[test_rgb_video_decoder.cpp]"){
 std::ifstream f(TEST_FILE); if(!f.good()) SKIP("missing sample"); RgbVideoDecoder dec; REQUIRE(dec.open(TEST_FILE)); RgbVideoDecoder::DecodedRgbFrame fr; REQUIRE(dec.readNextFrame(fr)); if(fr.is_valid && fr.hw_frame.frame) av_frame_free(&fr.hw_frame.frame); }
