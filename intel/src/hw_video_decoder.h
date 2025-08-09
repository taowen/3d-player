#pragma once
#include "mkv_stream_reader.h"
#include <memory>
#include <string>
#define WIN32_LEAN_AND_MEAN
#include <d3d11.h>
#include <wrl/client.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}

class HwVideoDecoder {
public:
	struct DecodedFrame { AVFrame* frame; bool is_valid; };
	HwVideoDecoder();
	~HwVideoDecoder();
	bool open(const std::string& filepath);
	bool readNextFrame(DecodedFrame& frame);
	bool isOpen() const;
	bool isEOF() const;
	void close();
	MKVStreamReader* getStreamReader() const;
	ID3D11Device* getD3D11Device() const;
	ID3D11DeviceContext* getD3D11DeviceContext() const;
private:
	std::unique_ptr<MKVStreamReader> stream_reader_;
	AVCodecContext* codec_context_;
	AVBufferRef* hw_device_ctx_;
	bool initializeHardwareDecoder();
	bool processPacket(AVPacket* packet, DecodedFrame& frame);
	void cleanup();
};
