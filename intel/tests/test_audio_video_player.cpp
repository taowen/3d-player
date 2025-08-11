#include <catch2/catch_test_macros.hpp>
#include "../src/audio_video_player.h"

TEST_CASE("AudioVideoPlayer open (intel)") {
	auto path = "test_data/sample_with_audio.mkv";
	AudioVideoPlayer avp;
	REQUIRE(avp.open(path));
	REQUIRE(avp.isOpen());
	REQUIRE(avp.hasAudio());
}
