#include "hw_video_decoder.h"
#include <iostream>

HwVideoDecoder::HwVideoDecoder()
    : stream_reader_(std::make_unique<MKVStreamReader>())
    , codec_context_(nullptr)
    , current_frame_index_(0)
    , hw_device_ctx_(nullptr)
    , hw_frames_ctx_(nullptr) {
    hw_frames_[0] = nullptr;
    hw_frames_[1] = nullptr;
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
    
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "Codec not found for codec_id: " << codecpar->codec_id << std::endl;
        return false;
    }
    
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
        std::cerr << "Failed to allocate codec context" << std::endl;
        return false;
    }
    
    if (avcodec_parameters_to_context(codec_context_, codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters to context" << std::endl;
        return false;
    }
    
    // Let FFmpeg create the D3D11VA device
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "Failed to create D3D11VA device context: " << ret << std::endl;
        return false;
    }
    
    codec_context_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    
    if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec" << std::endl;
        return false;
    }
    
    return true;
}

bool HwVideoDecoder::createHWFramesContext() {
    if (!hw_device_ctx_ || !codec_context_) {
        return false;
    }
    
    // Create hardware frames context
    hw_frames_ctx_ = av_hwframe_ctx_alloc(hw_device_ctx_);
    if (!hw_frames_ctx_) {
        std::cerr << "Failed to allocate hardware frames context" << std::endl;
        return false;
    }
    
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx_->data;
    frames_ctx->format = AV_PIX_FMT_D3D11;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;  // Common hardware decode format
    frames_ctx->width = codec_context_->width;
    frames_ctx->height = codec_context_->height;
    frames_ctx->initial_pool_size = 2;  // Double buffer, only need 2 frames
    
    int ret = av_hwframe_ctx_init(hw_frames_ctx_);
    if (ret < 0) {
        std::cerr << "Failed to initialize hardware frames context: " << ret << std::endl;
        av_buffer_unref(&hw_frames_ctx_);
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
    
    // First try to receive a frame to get correct width and height
    AVFrame* temp_frame = av_frame_alloc();
    if (!temp_frame) {
        std::cerr << "Failed to allocate temporary frame" << std::endl;
        return false;
    }
    
    ret = avcodec_receive_frame(codec_context_, temp_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_frame_free(&temp_frame);
        return false;
    } else if (ret < 0) {
        std::cerr << "Error receiving frame from decoder: " << ret << std::endl;
        av_frame_free(&temp_frame);
        return false;
    }
    
    // If hardware frames context hasn't been created yet, create it now
    if (!hw_frames_ctx_) {
        if (!createHWFramesContext()) {
            std::cerr << "Failed to create hardware frames context" << std::endl;
            av_frame_free(&temp_frame);
            return false;
        }
        
        // Allocate hardware frame buffers
        hw_frames_[0] = av_frame_alloc();
        hw_frames_[1] = av_frame_alloc();
        if (!hw_frames_[0] || !hw_frames_[1]) {
            std::cerr << "Failed to allocate frames" << std::endl;
            av_frame_free(&temp_frame);
            return false;
        }
        
        // Get memory from hardware frames context
        if (av_hwframe_get_buffer(hw_frames_ctx_, hw_frames_[0], 0) < 0 ||
            av_hwframe_get_buffer(hw_frames_ctx_, hw_frames_[1], 0) < 0) {
            std::cerr << "Failed to allocate hardware frame buffers" << std::endl;
            av_frame_free(&temp_frame);
            return false;
        }
    }
    
    // Now transfer frame data to hardware frame
    AVFrame* current_frame = hw_frames_[current_frame_index_];
    ret = av_hwframe_transfer_data(current_frame, temp_frame, 0);
    if (ret < 0) {
        std::cerr << "Failed to transfer frame to hardware: " << ret << std::endl;
        av_frame_free(&temp_frame);
        return false;
    }
    
    // Copy frame properties
    av_frame_copy_props(current_frame, temp_frame);
    
    av_frame_free(&temp_frame);
    
    return fillDecodedFrame(current_frame, frame);
}

bool HwVideoDecoder::fillDecodedFrame(AVFrame* frame, DecodedFrame& decoded_frame) {
    decoded_frame.frame = frame;
    decoded_frame.is_valid = true;
    
    // Rotate to next frame - implement double buffering
    current_frame_index_ = (current_frame_index_ + 1) % 2;
    
    return true;
}

void HwVideoDecoder::cleanup() {
    // Clean up double buffer frames
    for (int i = 0; i < 2; i++) {
        if (hw_frames_[i]) {
            av_frame_free(&hw_frames_[i]);
            hw_frames_[i] = nullptr;
        }
    }
    
    if (hw_frames_ctx_) {
        av_buffer_unref(&hw_frames_ctx_);
        hw_frames_ctx_ = nullptr;
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