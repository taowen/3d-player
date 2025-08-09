#include <catch2/catch_test_macros.hpp>
#include "../src/audio_video_player.h"
#include <fstream>
const std::string TEST_FILE="test_data/sample_with_audio.mkv";
TEST_CASE("AudioVideoPlayer open (intel)", "[test_audio_video_player.cpp]"){
 std::ifstream f(TEST_FILE); if(!f.good()) SKIP("missing sample"); AudioVideoPlayer avp; REQUIRE(avp.open(TEST_FILE)); }
