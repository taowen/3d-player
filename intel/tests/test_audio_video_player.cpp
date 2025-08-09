#include <catch2/catch_test_macros.hpp>
#include "../src/audio_video_player.h"
#include "test_utils.h"

TEST_CASE("AudioVideoPlayer open (intel)") {
	auto path = resolveTestMedia("sample_with_audio.mkv");
	REQUIRE(!path.empty());
	AudioVideoPlayer avp;
	REQUIRE(avp.open(path));
	REQUIRE(avp.isOpen());
	REQUIRE(avp.hasAudio());
}
