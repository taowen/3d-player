#include <catch2/catch_test_macros.hpp>
#include "../src/audio_decoder.h"
#include <fstream>
const std::string TEST_FILE="test_data/sample_with_audio.mkv";
TEST_CASE("AudioDecoder open and read (intel)", "[test_audio_decoder.cpp]"){
 std::ifstream f(TEST_FILE); if(!f.good()) SKIP("missing audio sample"); AudioDecoder dec; REQUIRE(dec.open(TEST_FILE)); AudioDecoder::DecodedFrame fr; int count=0; while(count<3 && dec.readNextFrame(fr)){ REQUIRE(fr.is_valid); av_frame_free(&fr.frame); count++; } REQUIRE(count>0); }
