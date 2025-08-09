// Intel variant: hardware decoder without CUDA adapter filtering
#include "hw_video_decoder.h"
#include <iostream>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixfmt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext_d3d11va.h>
}
inline std::string av_err2str_cpp_intel(int errnum){ char errbuf[AV_ERROR_MAX_STRING_SIZE]; av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE); return std::string(errbuf);} 
HwVideoDecoder::HwVideoDecoder():stream_reader_(std::make_unique<MKVStreamReader>()),codec_context_(nullptr),hw_device_ctx_(nullptr){}
HwVideoDecoder::~HwVideoDecoder(){ close(); }
bool HwVideoDecoder::open(const std::string& fp){ if(isOpen()) close(); if(!stream_reader_->open(fp)){ std::cerr<<"Failed to open file: "<<fp<<std::endl; return false;} if(!initializeHardwareDecoder()){ std::cerr<<"Failed to init hw decoder"<<std::endl; close(); return false;} return true; }
bool HwVideoDecoder::readNextFrame(DecodedFrame& f){ if(!isOpen()||isEOF()) return false; f.is_valid=false; AVPacket* pkt=av_packet_alloc(); if(!pkt) return false; while(stream_reader_->readNextPacket(pkt)){ if(stream_reader_->isVideoPacket(pkt)){ bool ok=processPacket(pkt,f); av_packet_unref(pkt); if(ok && f.is_valid){ av_packet_free(&pkt); return true;} } else { av_packet_unref(pkt);} } av_packet_free(&pkt); return false; }
bool HwVideoDecoder::isOpen() const { return stream_reader_ && stream_reader_->isOpen() && codec_context_ && hw_device_ctx_; }
bool HwVideoDecoder::isEOF() const { return stream_reader_ && stream_reader_->isEOF(); }
void HwVideoDecoder::close(){ cleanup(); if(stream_reader_) stream_reader_->close(); }
MKVStreamReader* HwVideoDecoder::getStreamReader() const { return stream_reader_.get(); }
ID3D11Device* HwVideoDecoder::getD3D11Device() const { if(!hw_device_ctx_) return nullptr; auto* d3d11_ctx=(AVD3D11VADeviceContext*)((AVHWDeviceContext*)hw_device_ctx_->data)->hwctx; return d3d11_ctx->device; }
ID3D11DeviceContext* HwVideoDecoder::getD3D11DeviceContext() const { if(!hw_device_ctx_) return nullptr; auto* d3d11_ctx=(AVD3D11VADeviceContext*)((AVHWDeviceContext*)hw_device_ctx_->data)->hwctx; return d3d11_ctx->device_context; }
bool HwVideoDecoder::initializeHardwareDecoder(){ AVCodecParameters* cp=stream_reader_->getVideoCodecParameters(); if(!cp){ std::cerr<<"No video codec parameters"<<std::endl; return false;} hw_device_ctx_=av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA); if(!hw_device_ctx_){ std::cerr<<"Alloc hw device ctx failed"<<std::endl; return false;} auto* device_ctx=(AVHWDeviceContext*)hw_device_ctx_->data; auto* d3d11va_ctx=(AVD3D11VADeviceContext*)device_ctx->hwctx; // Simple device creation (default adapter)
	HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&d3d11va_ctx->device,nullptr,&d3d11va_ctx->device_context);
	if(FAILED(hr)){ std::cerr<<"Create D3D11 device failed"<<std::endl; av_buffer_unref(&hw_device_ctx_); return false; }
	int ret = av_hwdevice_ctx_init(hw_device_ctx_); if(ret<0){ std::cerr<<"Init hw device ctx failed: "<<av_err2str_cpp_intel(ret)<<std::endl; av_buffer_unref(&hw_device_ctx_); return false; }
	const AVCodec* codec=nullptr; void* it=nullptr; const AVCodec* c; while((c=av_codec_iterate(&it))){ if(!av_codec_is_decoder(c) || c->id != cp->codec_id) continue; for(int j=0;;j++){ const AVCodecHWConfig* cfg = avcodec_get_hw_config(c,j); if(!cfg) break; if(cfg->device_type==AV_HWDEVICE_TYPE_D3D11VA){ codec=c; break;} } if(codec) break; }
	if(!codec){ std::cerr<<"No D3D11VA decoder for codec"<<std::endl; return false; }
	codec_context_=avcodec_alloc_context3(codec); if(!codec_context_) return false; if(avcodec_parameters_to_context(codec_context_, cp)<0) return false; codec_context_->hw_device_ctx = av_buffer_ref(hw_device_ctx_); codec_context_->extra_hw_frames = 40; codec_context_->get_format = [](AVCodecContext*, const enum AVPixelFormat* pix_fmts){ for(auto p=pix_fmts; *p!=AV_PIX_FMT_NONE; ++p){ if(*p==AV_PIX_FMT_D3D11) return *p; } for(auto p=pix_fmts; *p!=AV_PIX_FMT_NONE; ++p){ if(*p==AV_PIX_FMT_D3D11VA_VLD || *p==AV_PIX_FMT_DXVA2_VLD) return *p; } return AV_PIX_FMT_NONE; }; if(avcodec_open2(codec_context_, codec, nullptr)<0){ std::cerr<<"Open codec failed"<<std::endl; return false;} return true; }
bool HwVideoDecoder::processPacket(AVPacket* pkt, DecodedFrame& f){ int ret=avcodec_send_packet(codec_context_, pkt); if(ret<0) return false; AVFrame* cf=av_frame_alloc(); if(!cf) return false; ret=avcodec_receive_frame(codec_context_, cf); if(ret==AVERROR(EAGAIN)||ret==AVERROR_EOF){ av_frame_free(&cf); return false;} else if(ret<0){ av_frame_free(&cf); return false;} bool hw = (cf->format==AV_PIX_FMT_D3D11 || cf->format==AV_PIX_FMT_D3D11VA_VLD || cf->format==AV_PIX_FMT_DXVA2_VLD); if(!hw){ av_frame_free(&cf); return false;} f.frame=cf; f.is_valid=true; return true; }
void HwVideoDecoder::cleanup(){ if(codec_context_){ avcodec_free_context(&codec_context_); codec_context_=nullptr;} if(hw_device_ctx_){ av_buffer_unref(&hw_device_ctx_); hw_device_ctx_=nullptr; } }
