#include <catch2/catch_test_macros.hpp>
#include "../src/video_player.h"

TEST_CASE("VideoPlayer open (intel)") {
	auto path = "test_data/sample_hw.mkv";
	VideoPlayer vp;
	REQUIRE(vp.open(path));
	REQUIRE(vp.getWidth() > 0);
	REQUIRE(vp.getHeight() > 0);
}
