#include "mkv_stream_reader.h"
#include <iostream>

MKVStreamReader::MKVStreamReader() 
    : format_context_(nullptr), is_open_(false), is_eof_(false) {
}

MKVStreamReader::~MKVStreamReader() {
    close();
}

bool MKVStreamReader::open(const std::string& filepath) {
    if (is_open_) {
        close();
    }
    
    // Allocate format context
    format_context_ = avformat_alloc_context();
    if (!format_context_) {
        return false;
    }
    
    // Open file
    if (avformat_open_input(&format_context_, filepath.c_str(), nullptr, nullptr) < 0) {
        avformat_free_context(format_context_);
        format_context_ = nullptr;
        return false;
    }
    
    // Find stream information
    if (avformat_find_stream_info(format_context_, nullptr) < 0) {
        close();
        return false;
    }
    
    // Analyze streams and extract information
    if (!analyzeStreams()) {
        close();
        return false;
    }
    
    extractStreamInfo();
    is_open_ = true;
    is_eof_ = false;
    
    return true;
}

MKVStreamReader::StreamInfo MKVStreamReader::getStreamInfo() const {
    return stream_info_;
}

bool MKVStreamReader::readNextPacket(AVPacket* packet) {
    if (!is_open_ || is_eof_) {
        return false;
    }
    
    int ret = av_read_frame(format_context_, packet);
    if (ret < 0) {
        if (ret == AVERROR_EOF) {
            is_eof_ = true;
        }
        return false;
    }
    
    return true;
}

bool MKVStreamReader::isVideoPacket(const AVPacket* packet) const {
    if (!is_open_ || !packet) {
        return false;
    }
    return packet->stream_index == stream_info_.video_stream_index;
}

bool MKVStreamReader::isAudioPacket(const AVPacket* packet) const {
    if (!is_open_ || !packet) {
        return false;
    }
    return packet->stream_index == stream_info_.audio_stream_index;
}

bool MKVStreamReader::isOpen() const {
    return is_open_;
}

bool MKVStreamReader::isEOF() const {
    return is_eof_;
}

AVCodecParameters* MKVStreamReader::getVideoCodecParameters() const {
    if (!is_open_ || stream_info_.video_stream_index < 0) {
        return nullptr;
    }
    return format_context_->streams[stream_info_.video_stream_index]->codecpar;
}

AVCodecParameters* MKVStreamReader::getAudioCodecParameters() const {
    if (!is_open_ || stream_info_.audio_stream_index < 0) {
        return nullptr;
    }
    return format_context_->streams[stream_info_.audio_stream_index]->codecpar;
}

void MKVStreamReader::close() {
    if (format_context_) {
        avformat_close_input(&format_context_);
        format_context_ = nullptr;
    }
    
    is_open_ = false;
    is_eof_ = false;
    stream_info_ = StreamInfo{};
}

bool MKVStreamReader::analyzeStreams() {
    if (!format_context_) {
        return false;
    }
    
    stream_info_.video_stream_index = -1;
    stream_info_.audio_stream_index = -1;
    
    // Find video and audio streams
    for (unsigned int i = 0; i < format_context_->nb_streams; i++) {
        AVStream* stream = format_context_->streams[i];
        AVCodecParameters* codecpar = stream->codecpar;
        
        if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO && 
            stream_info_.video_stream_index == -1) {
            stream_info_.video_stream_index = i;
        } else if (codecpar->codec_type == AVMEDIA_TYPE_AUDIO && 
                   stream_info_.audio_stream_index == -1) {
            stream_info_.audio_stream_index = i;
        }
    }
    
    // At least need to find one video stream
    return stream_info_.video_stream_index >= 0;
}

void MKVStreamReader::extractStreamInfo() {
    if (!format_context_) {
        return;
    }
    
    // Extract video stream information
    if (stream_info_.video_stream_index >= 0) {
        AVStream* video_stream = format_context_->streams[stream_info_.video_stream_index];
        AVCodecParameters* video_codecpar = video_stream->codecpar;
        
        stream_info_.width = video_codecpar->width;
        stream_info_.height = video_codecpar->height;
        stream_info_.video_time_base = video_stream->time_base;
        stream_info_.video_bitrate = video_codecpar->bit_rate;
        
        // Get codec name
        const AVCodec* video_codec = avcodec_find_decoder(video_codecpar->codec_id);
        if (video_codec) {
            stream_info_.video_codec = video_codec->name;
        }
        
        // Calculate frame rate
        if (video_stream->avg_frame_rate.den != 0) {
            stream_info_.fps = av_q2d(video_stream->avg_frame_rate);
        } else if (video_stream->r_frame_rate.den != 0) {
            stream_info_.fps = av_q2d(video_stream->r_frame_rate);
        }
    }
    
    // Extract audio stream information
    if (stream_info_.audio_stream_index >= 0) {
        AVStream* audio_stream = format_context_->streams[stream_info_.audio_stream_index];
        AVCodecParameters* audio_codecpar = audio_stream->codecpar;
        
        stream_info_.audio_time_base = audio_stream->time_base;
        stream_info_.audio_bitrate = audio_codecpar->bit_rate;
        stream_info_.audio_sample_rate = audio_codecpar->sample_rate;
        stream_info_.audio_channels = audio_codecpar->ch_layout.nb_channels;
        
        // Get codec name
        const AVCodec* audio_codec = avcodec_find_decoder(audio_codecpar->codec_id);
        if (audio_codec) {
            stream_info_.audio_codec = audio_codec->name;
        }
    }
    
    // Extract duration information
    if (format_context_->duration != AV_NOPTS_VALUE) {
        stream_info_.duration = static_cast<double>(format_context_->duration) / AV_TIME_BASE;
    }
}