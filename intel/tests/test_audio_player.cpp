#include <catch2/catch_test_macros.hpp>
#include "../src/audio_player.h"
#include "test_utils.h"

TEST_CASE("AudioPlayer open + initialize (intel)") {
	auto path = resolveTestMedia("sample_with_audio.mkv");
	REQUIRE(!path.empty());
	AudioPlayer p;
	REQUIRE(p.open(path));
	p.initialize();
	REQUIRE(p.isOpen());
}
