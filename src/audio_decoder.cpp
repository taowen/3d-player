#include "audio_decoder.h"
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/samplefmt.h>
}


// Helper function for error string conversion on Windows
inline std::string av_err2str_cpp(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errbuf);
}


AudioDecoder::AudioDecoder()
    : stream_reader_(std::make_unique<MKVStreamReader>())
    , codec_context_(nullptr) {
}


AudioDecoder::~AudioDecoder() {
    close();
}


bool AudioDecoder::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    if (!stream_reader_->open(filepath)) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    if (!initializeAudioDecoder()) {
        std::cerr << "Failed to initialize audio decoder" << std::endl;
        close();
        return false;
    }
    
    return true;
}


bool AudioDecoder::readNextFrame(DecodedFrame& frame) {
    if (!isOpen() || isEOF()) {
        return false;
    }
    
    frame.is_valid = false;
    
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        return false;
    }
    
    while (stream_reader_->readNextPacket(packet)) {
        if (stream_reader_->isAudioPacket(packet)) {
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


bool AudioDecoder::isOpen() const {
    return stream_reader_->isOpen() && codec_context_ != nullptr;
}


bool AudioDecoder::isEOF() const {
    return stream_reader_->isEOF();
}


void AudioDecoder::close() {
    cleanup();
    
    if (stream_reader_) {
        stream_reader_->close();
    }
}


MKVStreamReader* AudioDecoder::getStreamReader() const {
    return stream_reader_.get();
}


bool AudioDecoder::initializeAudioDecoder() {
    AVCodecParameters* codecpar = stream_reader_->getAudioCodecParameters();
    if (!codecpar) {
        std::cerr << "No audio codec parameters" << std::endl;
        return false;
    }
    
    // 查找音频解码器
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "No audio decoder found for codec_id: " << codecpar->codec_id << std::endl;
        return false;
    }
    
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
        std::cerr << "Failed to allocate audio codec context" << std::endl;
        return false;
    }
    
    // 复制编解码器参数到上下文
    if (avcodec_parameters_to_context(codec_context_, codecpar) < 0) {
        std::cerr << "Failed to copy audio codec parameters to context" << std::endl;
        return false;
    }
    
    if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
        std::cerr << "Failed to open audio codec" << std::endl;
        return false;
    }
    
    return true;
}


bool AudioDecoder::processPacket(AVPacket* packet, DecodedFrame& frame) {
    int ret = avcodec_send_packet(codec_context_, packet);
    if (ret < 0) {
        std::cerr << "Error sending packet to audio decoder: " << av_err2str_cpp(ret) << std::endl;
        return false;
    }
    
    // 使用 FFmpeg 自动管理的帧缓冲区
    AVFrame* current_frame = av_frame_alloc();
    if (!current_frame) {
        std::cerr << "Failed to allocate audio frame" << std::endl;
        return false;
    }
    
    ret = avcodec_receive_frame(codec_context_, current_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_frame_free(&current_frame);
        return false;
    } else if (ret < 0) {
        std::cerr << "Error receiving frame from audio decoder: " << av_err2str_cpp(ret) << std::endl;
        av_frame_free(&current_frame);
        return false;
    }
    
    frame.frame = current_frame;
    frame.is_valid = true;
    
    return true;
}


void AudioDecoder::cleanup() {
    if (codec_context_) {
        avcodec_free_context(&codec_context_);
        codec_context_ = nullptr;
    }
} 