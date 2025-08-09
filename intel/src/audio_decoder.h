#pragma once

#include "mkv_stream_reader.h"
#include <memory>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
}

class AudioDecoder {
public:
	struct DecodedFrame { AVFrame* frame; bool is_valid; };
	AudioDecoder();
	~AudioDecoder();
	bool open(const std::string& filepath);
	bool readNextFrame(DecodedFrame& frame);
	bool isOpen() const; bool isEOF() const; void close();
	MKVStreamReader* getStreamReader() const;
private:
	std::unique_ptr<MKVStreamReader> stream_reader_;
	AVCodecContext* codec_context_{};
	bool initializeAudioDecoder();
	bool processPacket(AVPacket* packet, DecodedFrame& frame);
	void cleanup();
};
