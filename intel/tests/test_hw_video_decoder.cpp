#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include "test_utils.h"

TEST_CASE("HwVideoDecoder open and first frame (intel)") {
	auto path = resolveTestMedia("sample_hw.mkv");
	REQUIRE(!path.empty());
	HwVideoDecoder dec;
	REQUIRE(dec.open(path));
	HwVideoDecoder::DecodedFrame fr{nullptr,false};
	REQUIRE(dec.readNextFrame(fr));
	REQUIRE(fr.is_valid);
	if (fr.frame) {
		REQUIRE(fr.frame->width > 0);
		REQUIRE(fr.frame->height > 0);
		av_frame_free(&fr.frame);
	}
}
