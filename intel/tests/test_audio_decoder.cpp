#include <catch2/catch_test_macros.hpp>
#include "../src/audio_decoder.h"
#include "test_utils.h"

TEST_CASE("AudioDecoder open and read (intel)") {
	auto path = resolveTestMedia("sample_with_audio.mkv");
	REQUIRE(!path.empty());
	AudioDecoder dec;
	REQUIRE(dec.open(path));
	auto *sr = dec.getStreamReader();
	REQUIRE(sr != nullptr);
	auto info = sr->getStreamInfo();
	REQUIRE(info.audio_stream_index >= 0);
	REQUIRE(info.audio_sample_rate > 0);
	REQUIRE(info.audio_channels > 0);
	AudioDecoder::DecodedFrame fr{nullptr,false};
	int count = 0;
	while (count < 5 && dec.readNextFrame(fr)) {
		if (fr.is_valid && fr.frame) {
			REQUIRE(fr.frame->nb_samples > 0);
			av_frame_free(&fr.frame);
			count++;
		}
	}
	REQUIRE(count > 0);
}
