#include <catch2/catch_test_macros.hpp>
#include "../src/mkv_stream_reader.h"

TEST_CASE("MKVStreamReader basic (intel)") {
	auto path = "test_data/sample_hw.mkv";
	MKVStreamReader r;
	REQUIRE(r.open(path));
	auto info = r.getStreamInfo();
	REQUIRE(info.video_stream_index >= 0);
	REQUIRE(info.width > 0);
	REQUIRE(info.height > 0);
	REQUIRE(info.fps > 0.0);
	r.close();
}
