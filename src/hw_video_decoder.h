#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <string>
#include <memory>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_d3d11va.h>
}

#include "mkv_stream_reader.h"

using Microsoft::WRL::ComPtr;

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
    
    // D3D11 resource getters for external components
    ID3D11Device* getD3D11Device() const;
    ID3D11DeviceContext* getD3D11Context() const;

private:
    std::unique_ptr<MKVStreamReader> stream_reader_;
    
    AVCodecContext* codec_context_;
    AVFrame* hw_frames_[2];  // Double buffer frame array
    int current_frame_index_;  // Current frame index (0 or 1)
    bool frame_valid_[2];     // Track which frames contain valid data
    
    AVBufferRef* hw_device_ctx_;
    
    bool initializeFFmpegHWDecoder();
    bool processPacket(AVPacket* packet, DecodedFrame& frame);
    bool fillDecodedFrame(AVFrame* frame, DecodedFrame& decoded_frame, int buffer_index);
    int selectDecodeBuffer();  // Select which buffer to decode into for proper double buffering
    void cleanup();
};