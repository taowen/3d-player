#include <catch2/catch_test_macros.hpp>
#include "../src/mkv_stream_reader.h"
#include "test_utils.h"

TEST_CASE("MKVStreamReader basic (intel)") {
	auto path = resolveTestMedia("sample_hw.mkv");
	REQUIRE(!path.empty());
	MKVStreamReader r;
	REQUIRE(r.open(path));
	auto info = r.getStreamInfo();
	REQUIRE(info.video_stream_index >= 0);
	REQUIRE(info.width > 0);
	REQUIRE(info.height > 0);
	REQUIRE(info.fps > 0.0);
	r.close();
}
