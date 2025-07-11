#include "hw_video_decoder.h"
#include <iostream>

HwVideoDecoder::HwVideoDecoder()
    : stream_reader_(std::make_unique<MKVStreamReader>())
    , codec_context_(nullptr)
    , frame_(nullptr) {
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
    
    if (!initializeDecoder()) {
        std::cerr << "Failed to initialize decoder" << std::endl;
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

bool HwVideoDecoder::initializeDecoder() {
    AVCodecParameters* codecpar = stream_reader_->getVideoCodecParameters();
    if (!codecpar) {
        std::cerr << "No video codec parameters" << std::endl;
        return false;
    }
    
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "No decoder found for codec_id: " << codecpar->codec_id << std::endl;
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
    
    if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec" << std::endl;
        return false;
    }
    
    frame_ = av_frame_alloc();
    if (!frame_) {
        std::cerr << "Failed to allocate frame" << std::endl;
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
    
    av_frame_unref(frame_);
    
    ret = avcodec_receive_frame(codec_context_, frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return false;
    } else if (ret < 0) {
        std::cerr << "Error receiving frame from decoder: " << ret << std::endl;
        return false;
    }
    
    frame.frame = frame_;
    frame.is_valid = true;
    
    return true;
}

void HwVideoDecoder::cleanup() {
    if (frame_) {
        av_frame_free(&frame_);
        frame_ = nullptr;
    }
    
    if (codec_context_) {
        avcodec_free_context(&codec_context_);
        codec_context_ = nullptr;
    }
}