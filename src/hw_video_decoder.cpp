#include "hw_video_decoder.h"
#include <iostream>

HwVideoDecoder::HwVideoDecoder()
    : stream_reader_(std::make_unique<MKVStreamReader>())
    , codec_context_(nullptr)
    , current_frame_index_(0)
    , hw_device_ctx_(nullptr) {
    hw_frames_[0] = nullptr;
    hw_frames_[1] = nullptr;
    frame_valid_[0] = false;
    frame_valid_[1] = false;
}

HwVideoDecoder::~HwVideoDecoder() {
    close();
}

bool HwVideoDecoder::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    if (!stream_reader_->open(filepath)) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    if (!initializeFFmpegHWDecoder()) {
        std::cerr << "Failed to initialize FFmpeg hardware decoder" << std::endl;
        close();
        return false;
    }
    
    // Don't create hardware frames context here, create it on first decode
    // because some codecs need to decode some frames before determining the real video size
    
    current_frame_index_ = 0;
    frame_valid_[0] = false;
    frame_valid_[1] = false;
    
    return true;
}

bool HwVideoDecoder::readNextFrame(DecodedFrame& frame) {
    if (!isOpen() || isEOF()) {
        return false;
    }
    
    frame.is_valid = false;
    
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        return false;
    }
    
    while (stream_reader_->readNextPacket(packet)) {
        if (stream_reader_->isVideoPacket(packet)) {
            bool success = processPacket(packet, frame);
            av_packet_unref(packet);
            
            if (success && frame.is_valid) {
                av_packet_free(&packet);
                return true;
            }
        } else {
            av_packet_unref(packet);
        }
    }
    
    av_packet_free(&packet);
    return false;
}

bool HwVideoDecoder::isOpen() const {
    return stream_reader_->isOpen() && codec_context_ != nullptr;
}

bool HwVideoDecoder::isEOF() const {
    return stream_reader_->isEOF();
}

void HwVideoDecoder::close() {
    cleanup();
    
    if (stream_reader_) {
        stream_reader_->close();
    }
}

MKVStreamReader* HwVideoDecoder::getStreamReader() const {
    return stream_reader_.get();
}

bool HwVideoDecoder::initializeFFmpegHWDecoder() {
    AVCodecParameters* codecpar = stream_reader_->getVideoCodecParameters();
    if (!codecpar) {
        std::cerr << "No video codec parameters" << std::endl;
        return false;
    }
    
    // Use generic software decoder, hardware acceleration will be enabled via hwaccel
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "No decoder found for codec_id: " << codecpar->codec_id << std::endl;
        return false;
    }
    
    std::cerr << "Using decoder: " << codec->name << " with D3D11VA hardware acceleration" << std::endl;
    
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
        std::cerr << "Failed to allocate codec context" << std::endl;
        return false;
    }
    
    if (avcodec_parameters_to_context(codec_context_, codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters to context" << std::endl;
        return false;
    }
    
    // MUST use DirectX 11 hardware acceleration - no fallbacks allowed
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "FATAL ERROR: Failed to create D3D11VA device context: " << ret << std::endl;
        std::cerr << "DirectX 11 hardware acceleration is REQUIRED but not available!" << std::endl;
        return false;
    }
    std::cerr << "Successfully created D3D11VA device context" << std::endl;
    
    codec_context_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    
    // Enable D3D11VA hardware acceleration
    bool hardware_config_found = false;
    for (int i = 0; ; i++) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
        if (!config) {
            std::cerr << "No more hardware acceleration configs available" << std::endl;
            break;
        }
        
        std::cerr << "Hardware config " << i << ": device_type=" << config->device_type 
                  << ", methods=" << config->methods 
                  << ", pix_fmt=" << config->pix_fmt << std::endl;
        
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == AV_HWDEVICE_TYPE_D3D11VA) {
            std::cerr << "Found D3D11VA hardware acceleration config with pix_fmt=" << config->pix_fmt << std::endl;
            hardware_config_found = true;
            codec_context_->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
                const enum AVPixelFormat* p;
                
                // Print available formats for debugging
                std::cerr << "Available pixel formats: ";
                for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
                    std::cerr << *p << " ";
                }
                std::cerr << std::endl;
                
                // MUST find D3D11 hardware format - no fallbacks allowed
                for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
                    if (*p == AV_PIX_FMT_D3D11) {
                        std::cerr << "✓ Successfully selected AV_PIX_FMT_D3D11 for DirectX 11 hardware acceleration!" << std::endl;
                        return *p;
                    }
                    if (*p == AV_PIX_FMT_D3D11VA_VLD) {
                        std::cerr << "✓ Successfully selected AV_PIX_FMT_D3D11VA_VLD for DirectX 11 hardware acceleration!" << std::endl;
                        return *p;
                    }
                }
                
                // FATAL ERROR: DirectX 11 hardware acceleration is required but not available
                std::cerr << "FATAL ERROR: DirectX 11 hardware acceleration is REQUIRED but AV_PIX_FMT_D3D11 or AV_PIX_FMT_D3D11VA_VLD not available!" << std::endl;
                std::cerr << "Available formats do not include DirectX 11 hardware formats." << std::endl;
                return AV_PIX_FMT_NONE;  // This will cause decoding to fail
            };
            break;
        }
    }
    
    if (!hardware_config_found) {
        std::cerr << "FATAL ERROR: No D3D11VA hardware acceleration config found for this codec!" << std::endl;
        std::cerr << "DirectX 11 hardware acceleration is REQUIRED but not supported by this codec." << std::endl;
        return false;
    }
    
    if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec" << std::endl;
        return false;
    }
    
    // Allocate frame buffers for double buffering
    hw_frames_[0] = av_frame_alloc();
    hw_frames_[1] = av_frame_alloc();
    if (!hw_frames_[0] || !hw_frames_[1]) {
        std::cerr << "Failed to allocate frames" << std::endl;
        return false;
    }
    
    return true;
}



bool HwVideoDecoder::processPacket(AVPacket* packet, DecodedFrame& frame) {
    int ret = avcodec_send_packet(codec_context_, packet);
    if (ret < 0) {
        std::cerr << "Error sending packet to decoder: " << ret << std::endl;
        return false;
    }
    
    // Select the buffer to decode into
    // For proper double buffering, we need to choose the buffer that won't 
    // invalidate the most recently returned frame
    int decode_buffer_index = selectDecodeBuffer();
    
    // Clear the frame before receiving new data
    AVFrame* decode_frame = hw_frames_[decode_buffer_index];
    av_frame_unref(decode_frame);
    
    // Receive frame from hardware decoder
    ret = avcodec_receive_frame(codec_context_, decode_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return false;
    } else if (ret < 0) {
        std::cerr << "Error receiving frame from decoder: " << ret << std::endl;
        return false;
    }
    
    // Debug: Print frame format
    std::cerr << "Decoded frame format: " << decode_frame->format << std::endl;
    
    // Verify DirectX 11 hardware decoding is being used
    if (decode_frame->format == AV_PIX_FMT_D3D11) {
        std::cerr << "✅ SUCCESS: DirectX 11 hardware decoding with AV_PIX_FMT_D3D11!" << std::endl;
    } else if (decode_frame->format == AV_PIX_FMT_D3D11VA_VLD) {
        std::cerr << "✅ SUCCESS: DirectX 11 hardware decoding with AV_PIX_FMT_D3D11VA_VLD!" << std::endl;
    } else {
        std::cerr << "❌ FATAL ERROR: Expected DirectX 11 hardware format but got: " << decode_frame->format << std::endl;
        std::cerr << "❌ This should not happen if DirectX 11 hardware acceleration is properly configured!" << std::endl;
        return false;  // Fail immediately if not using DirectX 11 hardware decoding
    }
    
    return fillDecodedFrame(decode_frame, frame, decode_buffer_index);
}

bool HwVideoDecoder::fillDecodedFrame(AVFrame* frame, DecodedFrame& decoded_frame, int buffer_index) {
    decoded_frame.frame = frame;
    decoded_frame.is_valid = true;
    
    // Mark this buffer as valid and update current frame index
    frame_valid_[buffer_index] = true;
    current_frame_index_ = buffer_index;
    
    return true;
}

int HwVideoDecoder::selectDecodeBuffer() {
    // Double buffering strategy:
    // - Keep the most recently returned frame valid
    // - Use the other buffer for new decoding
    
    // If both buffers are valid, use the one that's not the current frame
    if (frame_valid_[0] && frame_valid_[1]) {
        return (current_frame_index_ == 0) ? 1 : 0;
    }
    
    // If only one buffer is valid, use the other one
    if (frame_valid_[0]) {
        return 1;
    }
    if (frame_valid_[1]) {
        return 0;
    }
    
    // If neither buffer is valid, start with buffer 0
    return 0;
}

void HwVideoDecoder::cleanup() {
    // Clean up double buffer frames
    for (int i = 0; i < 2; i++) {
        if (hw_frames_[i]) {
            av_frame_free(&hw_frames_[i]);
            hw_frames_[i] = nullptr;
        }
        frame_valid_[i] = false;
    }
    

    
    if (codec_context_) {
        avcodec_free_context(&codec_context_);
        codec_context_ = nullptr;
    }
    
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
    
    current_frame_index_ = 0;
}

ID3D11Device* HwVideoDecoder::getD3D11Device() const {
    if (!hw_device_ctx_) {
        return nullptr;
    }
    
    AVHWDeviceContext* device_ctx = (AVHWDeviceContext*)hw_device_ctx_->data;
    if (device_ctx->type != AV_HWDEVICE_TYPE_D3D11VA) {
        return nullptr;
    }
    
    AVD3D11VADeviceContext* d3d11_ctx = (AVD3D11VADeviceContext*)device_ctx->hwctx;
    return d3d11_ctx->device;
}

ID3D11DeviceContext* HwVideoDecoder::getD3D11Context() const {
    if (!hw_device_ctx_) {
        return nullptr;
    }
    
    AVHWDeviceContext* device_ctx = (AVHWDeviceContext*)hw_device_ctx_->data;
    if (device_ctx->type != AV_HWDEVICE_TYPE_D3D11VA) {
        return nullptr;
    }
    
    AVD3D11VADeviceContext* d3d11_ctx = (AVD3D11VADeviceContext*)device_ctx->hwctx;
    return d3d11_ctx->device_context;
}