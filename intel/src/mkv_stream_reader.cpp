#include "mkv_stream_reader.h"
#include <iostream>

MKVStreamReader::MKVStreamReader() 
	: format_context_(nullptr), is_open_(false), is_eof_(false) {}

MKVStreamReader::~MKVStreamReader() { close(); }

bool MKVStreamReader::open(const std::string& filepath) {
	if (is_open_) { close(); }
	format_context_ = avformat_alloc_context();
	if (!format_context_) return false;
	if (avformat_open_input(&format_context_, filepath.c_str(), nullptr, nullptr) < 0) {
		avformat_free_context(format_context_); format_context_ = nullptr; return false; }
	if (avformat_find_stream_info(format_context_, nullptr) < 0) { close(); return false; }
	if (!analyzeStreams()) { close(); return false; }
	extractStreamInfo();
	is_open_ = true; is_eof_ = false; return true; }

MKVStreamReader::StreamInfo MKVStreamReader::getStreamInfo() const { return stream_info_; }

bool MKVStreamReader::readNextPacket(AVPacket* packet) {
	if (!is_open_ || is_eof_) return false; int ret = av_read_frame(format_context_, packet);
	if (ret < 0) { if (ret == AVERROR_EOF) is_eof_ = true; return false; } return true; }

bool MKVStreamReader::isVideoPacket(const AVPacket* packet) const { return is_open_ && packet && packet->stream_index == stream_info_.video_stream_index; }
bool MKVStreamReader::isAudioPacket(const AVPacket* packet) const { return is_open_ && packet && packet->stream_index == stream_info_.audio_stream_index; }
bool MKVStreamReader::isOpen() const { return is_open_; }
bool MKVStreamReader::isEOF() const { return is_eof_; }

AVCodecParameters* MKVStreamReader::getVideoCodecParameters() const { if (!is_open_ || stream_info_.video_stream_index < 0) return nullptr; return format_context_->streams[stream_info_.video_stream_index]->codecpar; }
AVCodecParameters* MKVStreamReader::getAudioCodecParameters() const { if (!is_open_ || stream_info_.audio_stream_index < 0) return nullptr; return format_context_->streams[stream_info_.audio_stream_index]->codecpar; }

void MKVStreamReader::close() { if (format_context_) { avformat_close_input(&format_context_); format_context_ = nullptr; } is_open_ = false; is_eof_ = false; stream_info_ = StreamInfo{}; }

bool MKVStreamReader::analyzeStreams() { if (!format_context_) return false; stream_info_.video_stream_index = -1; stream_info_.audio_stream_index = -1; for (unsigned int i=0;i<format_context_->nb_streams;i++){ AVStream* s = format_context_->streams[i]; AVCodecParameters* cp = s->codecpar; if (cp->codec_type==AVMEDIA_TYPE_VIDEO && stream_info_.video_stream_index==-1) stream_info_.video_stream_index = i; else if (cp->codec_type==AVMEDIA_TYPE_AUDIO && stream_info_.audio_stream_index==-1) stream_info_.audio_stream_index = i; } return stream_info_.video_stream_index >= 0; }

void MKVStreamReader::extractStreamInfo() { if (!format_context_) return; if (stream_info_.video_stream_index>=0){ AVStream* vs = format_context_->streams[stream_info_.video_stream_index]; AVCodecParameters* vcp = vs->codecpar; stream_info_.width = vcp->width; stream_info_.height = vcp->height; stream_info_.video_time_base = vs->time_base; stream_info_.video_bitrate = vcp->bit_rate; const AVCodec* codec = avcodec_find_decoder(vcp->codec_id); if (codec) stream_info_.video_codec = codec->name; if (vs->avg_frame_rate.den!=0) stream_info_.fps = av_q2d(vs->avg_frame_rate); else if (vs->r_frame_rate.den!=0) stream_info_.fps = av_q2d(vs->r_frame_rate); }
 if (stream_info_.audio_stream_index>=0){ AVStream* as = format_context_->streams[stream_info_.audio_stream_index]; AVCodecParameters* acp = as->codecpar; stream_info_.audio_time_base = as->time_base; stream_info_.audio_bitrate = acp->bit_rate; stream_info_.audio_sample_rate = acp->sample_rate; stream_info_.audio_channels = acp->ch_layout.nb_channels; const AVCodec* codec = avcodec_find_decoder(acp->codec_id); if (codec) stream_info_.audio_codec = codec->name; }
 if (format_context_->duration != AV_NOPTS_VALUE) stream_info_.duration = (double)format_context_->duration / AV_TIME_BASE; }
