// Intel variant: standalone copy of MKVStreamReader (no stereo specific changes)
#pragma once

extern "C" {
	#include <libavformat/avformat.h>
	#include <libavcodec/avcodec.h>
}
#include <string>

class MKVStreamReader {
public:
	struct StreamInfo {
		int video_stream_index = -1;
		int audio_stream_index = -1;
		std::string video_codec;
		std::string audio_codec;
		int width = 0;
		int height = 0;
		double duration = 0.0;
		AVRational video_time_base = {0, 1};
		AVRational audio_time_base = {0, 1};
		double fps = 0.0;
		int64_t video_bitrate = 0;
		int64_t audio_bitrate = 0;
		int audio_sample_rate = 0;
		int audio_channels = 0;
	};

	MKVStreamReader();
	~MKVStreamReader();
	bool open(const std::string& filepath);
	StreamInfo getStreamInfo() const;
	bool readNextPacket(AVPacket* packet);
	bool isVideoPacket(const AVPacket* packet) const;
	bool isAudioPacket(const AVPacket* packet) const;
	bool isOpen() const;
	bool isEOF() const;
	AVCodecParameters* getVideoCodecParameters() const;
	AVCodecParameters* getAudioCodecParameters() const;
	void close();
private:
	AVFormatContext* format_context_;
	StreamInfo stream_info_;
	bool is_open_;
	bool is_eof_;
	bool analyzeStreams();
	void extractStreamInfo();
};
