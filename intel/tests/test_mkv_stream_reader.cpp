#include <catch2/catch_test_macros.hpp>
#include "../src/mkv_stream_reader.h"
#include <fstream>
const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";
TEST_CASE("MKVStreamReader basic (intel)", "[test_mkv_stream_reader.cpp]"){
 MKVStreamReader r; std::ifstream f(TEST_MKV_FILE); if(!f.good()) SKIP("missing file"); REQUIRE(r.open(TEST_MKV_FILE)); auto info=r.getStreamInfo(); REQUIRE(info.video_stream_index>=0); r.close(); }
