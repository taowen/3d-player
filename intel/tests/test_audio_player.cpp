#include <catch2/catch_test_macros.hpp>
#include "../src/audio_player.h"
#include <fstream>
const std::string TEST_FILE="test_data/sample_with_audio.mkv";
TEST_CASE("AudioPlayer open (intel)", "[test_audio_player.cpp]"){
 std::ifstream f(TEST_FILE); if(!f.good()) SKIP("missing audio sample"); AudioPlayer p; REQUIRE(p.open(TEST_FILE)); }
