#include "hw_video_decoder.h"
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
#include <libavutil/pixdesc.h>
}

// Helper function for error string conversion on Windows
inline std::string av_err2str_cpp(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errbuf);
}

HwVideoDecoder::HwVideoDecoder()
    : stream_reader_(std::make_unique<MKVStreamReader>())
    , codec_context_(nullptr)
    , hw_device_ctx_(nullptr) {
    // FFmpeg automatically manages buffers, no manual initialization needed
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
    
    if (!initializeHardwareDecoder()) {
        std::cerr << "Failed to initialize hardware decoder" << std::endl;
        close();
        return false;
    }
    
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
    return stream_reader_->isOpen() && codec_context_ != nullptr && hw_device_ctx_ != nullptr;
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

ID3D11Device* HwVideoDecoder::getD3D11Device() const {
    if (!hw_device_ctx_) {
        return nullptr;
    }
    
    AVD3D11VADeviceContext* d3d11_ctx = (AVD3D11VADeviceContext*)((AVHWDeviceContext*)hw_device_ctx_->data)->hwctx;
    return d3d11_ctx->device;
}

ID3D11DeviceContext* HwVideoDecoder::getD3D11DeviceContext() const {
    if (!hw_device_ctx_) {
        return nullptr;
    }
    
    AVD3D11VADeviceContext* d3d11_ctx = (AVD3D11VADeviceContext*)((AVHWDeviceContext*)hw_device_ctx_->data)->hwctx;
    return d3d11_ctx->device_context;
}

bool HwVideoDecoder::initializeHardwareDecoder() {
    AVCodecParameters* codecpar = stream_reader_->getVideoCodecParameters();
    if (!codecpar) {
        std::cerr << "No video codec parameters" << std::endl;
        return false;
    }
    
    // Create D3D11VA hardware device context
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "Failed to create D3D11VA device context: " << av_err2str_cpp(ret) << std::endl;
        return false;
    }
    
    // Find D3D11VA hardware decoder dynamically
    const AVCodec* codec = nullptr;
    void* i = nullptr;
    const AVCodec* c;
    
    // Iterate through all available codecs to find D3D11VA hardware decoder
    while ((c = av_codec_iterate(&i))) {
        // Check if this is a decoder with matching codec ID
        if (!av_codec_is_decoder(c) || c->id != codecpar->codec_id) {
            continue;
        }
        
        // Check hardware configurations for D3D11VA support
        for (int j = 0;; j++) {
            const AVCodecHWConfig* config = avcodec_get_hw_config(c, j);
            if (!config) {
                break; // No more hardware configs for this codec
            }
            
            // Check if this codec supports D3D11VA
            if (config->device_type == AV_HWDEVICE_TYPE_D3D11VA) {
                codec = c;
                std::cerr << "Found D3D11VA decoder: " << codec->name << " for codec_id: " << codecpar->codec_id << std::endl;
                break;
            }
        }
        
        if (codec) {
            break; // Found D3D11VA decoder
        }
    }
    
    if (!codec) {
        std::cerr << "No D3D11VA decoder found for codec_id: " << codecpar->codec_id << std::endl;
        std::cerr << "Make sure DirectX 11 hardware acceleration is available on your system" << std::endl;
        return false;
    }
    
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
        std::cerr << "Failed to allocate codec context" << std::endl;
        return false;
    }
    
    // Copy codec parameters to context first
    if (avcodec_parameters_to_context(codec_context_, codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters to context" << std::endl;
        return false;
    }
    
    // Set hardware device context - FFmpeg will auto-create frames context and buffer pool
    codec_context_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    
    // 设置硬件解码的格式回调函数，只允许硬件解码
    codec_context_->get_format = [](AVCodecContext* /*ctx*/, const enum AVPixelFormat* pix_fmts) -> enum AVPixelFormat {
        const enum AVPixelFormat* p;
        
        // 首选 D3D11 格式
        for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
            if (*p == AV_PIX_FMT_D3D11) {
                return *p;
            }
        }
        
        // 如果没有 D3D11，尝试其他硬件格式
        for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
            if (*p == AV_PIX_FMT_D3D11VA_VLD || *p == AV_PIX_FMT_DXVA2_VLD || 
                *p == AV_PIX_FMT_CUDA || *p == AV_PIX_FMT_VULKAN) {
                return *p;
            }
        }
        
        // 如果没有硬件格式，返回 NONE 表示失败
        std::cerr << "No hardware pixel format available, hardware decoding required" << std::endl;
        return AV_PIX_FMT_NONE;
    };
    
    if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec" << std::endl;
        return false;
    }
    
    return true;
}

bool HwVideoDecoder::processPacket(AVPacket* packet, DecodedFrame& frame) {
    int ret = avcodec_send_packet(codec_context_, packet);
    if (ret < 0) {
        std::cerr << "Error sending packet to decoder: " << av_err2str_cpp(ret) << std::endl;
        return false;
    }
    
    // Use FFmpeg automatically managed frame buffers
    AVFrame* current_frame = av_frame_alloc();
    if (!current_frame) {
        std::cerr << "Failed to allocate frame" << std::endl;
        return false;
    }
    
    ret = avcodec_receive_frame(codec_context_, current_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_frame_free(&current_frame);
        return false;
    } else if (ret < 0) {
        std::cerr << "Error receiving frame from decoder: " << av_err2str_cpp(ret) << std::endl;
        av_frame_free(&current_frame);
        return false;
    }
    
    // 检查是否是硬件解码格式，只允许硬件解码
    bool is_hardware_format = (current_frame->format == AV_PIX_FMT_D3D11 ||
                               current_frame->format == AV_PIX_FMT_D3D11VA_VLD ||
                               current_frame->format == AV_PIX_FMT_DXVA2_VLD ||
                               current_frame->format == AV_PIX_FMT_CUDA ||
                               current_frame->format == AV_PIX_FMT_VULKAN);
    
    if (!is_hardware_format) {
        std::cerr << "Frame is not hardware decoded (format: " << current_frame->format << "), hardware decoding required" << std::endl;
        av_frame_free(&current_frame);
        return false;
    }
    
    frame.frame = current_frame;
    frame.is_valid = true;
    
    return true;
}

void HwVideoDecoder::cleanup() {
    if (codec_context_) {
        avcodec_free_context(&codec_context_);
        codec_context_ = nullptr;
    }
    
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
}