#include <catch2/catch_test_macros.hpp>
#include "../src/video_player.h"
#include "test_utils.h"

TEST_CASE("VideoPlayer open (intel)") {
	auto path = resolveTestMedia("sample_hw.mkv");
	REQUIRE(!path.empty());
	VideoPlayer vp;
	REQUIRE(vp.open(path));
	REQUIRE(vp.getWidth() > 0);
	REQUIRE(vp.getHeight() > 0);
}
