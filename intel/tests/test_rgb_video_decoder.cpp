#include <catch2/catch_test_macros.hpp>
#include "../src/rgb_video_decoder.h"

TEST_CASE("RgbVideoDecoder first frame (intel)") {
	auto path = "test_data/sample_hw.mkv";
	RgbVideoDecoder dec;
	REQUIRE(dec.open(path));
	RgbVideoDecoder::DecodedRgbFrame fr{{nullptr,false},nullptr,false};
	REQUIRE(dec.readNextFrame(fr));
	REQUIRE(fr.is_valid);
	REQUIRE(fr.rgb_texture != nullptr);
	if (fr.hw_frame.frame) av_frame_free(&fr.hw_frame.frame);
}
