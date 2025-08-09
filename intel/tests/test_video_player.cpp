#include <catch2/catch_test_macros.hpp>
#include "../src/video_player.h"
#include <fstream>
const std::string TEST_FILE="test_data/sample_hw.mkv";
TEST_CASE("VideoPlayer open (intel)", "[test_video_player.cpp]"){
 std::ifstream f(TEST_FILE); if(!f.good()) SKIP("missing sample"); VideoPlayer vp; REQUIRE(vp.open(TEST_FILE)); }
