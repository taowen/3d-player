#pragma once

#include <string>
#include <memory>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
}

#include "mkv_stream_reader.h"

class HwVideoDecoder {
public:
    struct DecodedFrame {
        AVFrame* frame;
        bool is_valid;
        
        DecodedFrame() : frame(nullptr), is_valid(false) {}
    };

    HwVideoDecoder();
    ~HwVideoDecoder();

    bool open(const std::string& filepath);
    bool readNextFrame(DecodedFrame& frame);
    
    bool isOpen() const;
    bool isEOF() const;
    void close();
    
    MKVStreamReader* getStreamReader() const;

private:
    std::unique_ptr<MKVStreamReader> stream_reader_;
    AVCodecContext* codec_context_;
    AVFrame* frame_;
    
    bool initializeDecoder();
    bool processPacket(AVPacket* packet, DecodedFrame& frame);
    void cleanup();
};