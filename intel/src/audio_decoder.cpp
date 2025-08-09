#include "audio_decoder.h"
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
}

AudioDecoder::AudioDecoder():stream_reader_(std::make_unique<MKVStreamReader>()){}
AudioDecoder::~AudioDecoder(){ close(); }
bool AudioDecoder::open(const std::string& filepath){ if(isOpen()) close(); if(!stream_reader_->open(filepath)){ std::cerr<<"Failed to open file for audio: "<<filepath<<std::endl; return false;} if(!initializeAudioDecoder()){ std::cerr<<"Init audio decoder failed"<<std::endl; close(); return false;} return true; }
bool AudioDecoder::initializeAudioDecoder(){ AVCodecParameters* cp = stream_reader_->getAudioCodecParameters(); if(!cp){ return false;} const AVCodec* codec = avcodec_find_decoder(cp->codec_id); if(!codec) return false; codec_context_ = avcodec_alloc_context3(codec); if(!codec_context_) return false; if(avcodec_parameters_to_context(codec_context_, cp) < 0) return false; if(avcodec_open2(codec_context_, codec, nullptr) < 0) return false; return true; }
bool AudioDecoder::readNextFrame(DecodedFrame& frame){ if(!isOpen()||isEOF()) return false; frame.is_valid=false; AVPacket* pkt=av_packet_alloc(); if(!pkt) return false; while(stream_reader_->readNextPacket(pkt)){ if(stream_reader_->isAudioPacket(pkt)){ bool ok=processPacket(pkt, frame); av_packet_unref(pkt); if(ok && frame.is_valid){ av_packet_free(&pkt); return true; } } else { av_packet_unref(pkt); } } av_packet_free(&pkt); return false; }
bool AudioDecoder::processPacket(AVPacket* packet, DecodedFrame& f){ int ret=avcodec_send_packet(codec_context_, packet); if(ret<0) return false; AVFrame* frm = av_frame_alloc(); if(!frm) return false; ret=avcodec_receive_frame(codec_context_, frm); if(ret==AVERROR(EAGAIN)||ret==AVERROR_EOF){ av_frame_free(&frm); return false; } else if(ret<0){ av_frame_free(&frm); return false; } f.frame=frm; f.is_valid=true; return true; }
bool AudioDecoder::isOpen() const { return stream_reader_ && stream_reader_->isOpen() && codec_context_; }
bool AudioDecoder::isEOF() const { return stream_reader_ && stream_reader_->isEOF(); }
void AudioDecoder::close(){ if(codec_context_){ avcodec_free_context(&codec_context_); codec_context_=nullptr;} if(stream_reader_) stream_reader_->close(); }
void AudioDecoder::cleanup(){ if(codec_context_){ avcodec_free_context(&codec_context_); codec_context_=nullptr;} }
MKVStreamReader* AudioDecoder::getStreamReader() const { return stream_reader_.get(); }
