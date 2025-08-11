#include <catch2/catch_test_macros.hpp>
#include "../src/audio_player.h"

TEST_CASE("AudioPlayer open + initialize (intel)") {
	auto path = "test_data/sample_with_audio.mkv";
	AudioPlayer p;
	REQUIRE(p.open(path));
	p.initialize();
	REQUIRE(p.isOpen());
}
