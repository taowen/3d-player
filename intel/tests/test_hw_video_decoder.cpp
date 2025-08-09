#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <fstream>
const std::string TEST_FILE="test_data/sample_hw.mkv";
TEST_CASE("HwVideoDecoder open and first frame (intel)", "[test_hw_video_decoder.cpp]"){
 std::ifstream f(TEST_FILE); if(!f.good()) SKIP("missing sample"); HwVideoDecoder dec; REQUIRE(dec.open(TEST_FILE)); HwVideoDecoder::DecodedFrame fr; REQUIRE(dec.readNextFrame(fr)); if(fr.is_valid && fr.frame) av_frame_free(&fr.frame); }
