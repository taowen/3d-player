#include "video_player.h"
#include <d3dcompiler.h>
#include <d3d11.h>
#include <iostream>
extern "C" {
#include <libavutil/rational.h>
}

VideoPlayer::VideoPlayer():rgb_decoder_(std::make_unique<RgbVideoDecoder>()){}
VideoPlayer::~VideoPlayer(){ close(); }
bool VideoPlayer::open(const std::string& fp){ if(isOpen()) close(); if(!rgb_decoder_->open(fp)){ std::cerr<<"Failed to open rgb decoder"<<std::endl; return false;} return true; }
bool VideoPlayer::setRenderTarget(ComPtr<ID3D11Texture2D> rt){ if(!isOpen()) return false; render_target_view_.Reset(); render_target_texture_.Reset(); render_target_swapchain_.Reset(); if(!initializeRenderTarget(rt)) return false; if(!preloadNextFrame()) return false; return true; }
bool VideoPlayer::setRenderTarget(ComPtr<IDXGISwapChain> sc){ if(!isOpen()) return false; render_target_view_.Reset(); render_target_texture_.Reset(); render_target_swapchain_.Reset(); if(!initializeRenderTarget(sc)) return false; if(!preloadNextFrame()) return false; return true; }
void VideoPlayer::onTimer(double t){ if(!isReady()) return; std::lock_guard<std::mutex> lk(frame_mutex_); bool need=false; while(!frame_buffer_.empty()){ auto &front=frame_buffer_.front(); double frame_time=0.0; if(front.frame.hw_frame.frame && front.frame.hw_frame.frame->pts!=AV_NOPTS_VALUE){ AVRational tb = rgb_decoder_->getHwDecoder()->getStreamReader()->getStreamInfo().video_time_base; frame_time = front.frame.hw_frame.frame->pts * av_q2d(tb);} if(frame_time <= t){ current_frame_texture_=front.d3d_texture; current_frame_time_=frame_time; if(front.frame.hw_frame.frame) av_frame_free(&front.frame.hw_frame.frame); frame_buffer_.pop(); need=true; preloadNextFrame(); } else break; } if(need && current_frame_texture_) renderFrame(current_frame_texture_); }
ComPtr<ID3D11Texture2D> VideoPlayer::getCurrentFrame() const { std::lock_guard<std::mutex> lk(frame_mutex_); return current_frame_texture_; }
void VideoPlayer::close(){ std::lock_guard<std::mutex> lk(frame_mutex_); while(!frame_buffer_.empty()){ auto &f=frame_buffer_.front(); if(f.frame.hw_frame.frame) av_frame_free(&f.frame.hw_frame.frame); frame_buffer_.pop(); } current_frame_texture_.Reset(); current_frame_time_=0.0; render_target_view_.Reset(); render_target_texture_.Reset(); render_target_swapchain_.Reset(); fullscreen_vs_.Reset(); fullscreen_ps_.Reset(); texture_sampler_.Reset(); blend_state_.Reset(); raster_state_.Reset(); if(rgb_decoder_) rgb_decoder_->close(); }
bool VideoPlayer::isOpen() const { return rgb_decoder_ && rgb_decoder_->isOpen(); }
bool VideoPlayer::isReady() const { return isOpen() && render_target_view_; }
bool VideoPlayer::isEOF() const { return rgb_decoder_ && rgb_decoder_->isEOF(); }
RgbVideoDecoder* VideoPlayer::getRgbDecoder() const { return rgb_decoder_.get(); }
ID3D11Device* VideoPlayer::getD3D11Device() const { if(!isOpen()) return nullptr; return rgb_decoder_->getHwDecoder()->getD3D11Device(); }
ID3D11DeviceContext* VideoPlayer::getD3D11DeviceContext() const { if(!isOpen()) return nullptr; ID3D11Device* d=getD3D11Device(); if(!d) return nullptr; ID3D11DeviceContext* ctx=nullptr; d->GetImmediateContext(&ctx); return ctx; }
bool VideoPlayer::preloadNextFrame(){ if(!rgb_decoder_) return false; if(frame_buffer_.size()>=3) return true; RgbVideoDecoder::DecodedRgbFrame f; if(!rgb_decoder_->readNextFrame(f)) return false; DecodedRgbFrameWrapper w{f,f.rgb_texture}; frame_buffer_.push(w); return true; }
bool VideoPlayer::initializeRenderTarget(ComPtr<ID3D11Texture2D> rt){ if(!rt) return false; render_target_type_=RenderTargetType::TEXTURE; render_target_texture_=rt; ID3D11Device* dev=getD3D11Device(); if(!dev) return false; D3D11_RENDER_TARGET_VIEW_DESC d={}; d.Format=DXGI_FORMAT_B8G8R8A8_UNORM; d.ViewDimension=D3D11_RTV_DIMENSION_TEXTURE2D; d.Texture2D.MipSlice=0; if(FAILED(dev->CreateRenderTargetView(rt.Get(), &d,&render_target_view_))) return false; return true; }
bool VideoPlayer::initializeRenderTarget(ComPtr<IDXGISwapChain> sc){ if(!sc) return false; render_target_type_=RenderTargetType::SWAPCHAIN; render_target_swapchain_=sc; ID3D11Device* dev=getD3D11Device(); if(!dev) return false; ComPtr<ID3D11Texture2D> bb; if(FAILED(sc->GetBuffer(0,__uuidof(ID3D11Texture2D),(void**)&bb))) return false; if(FAILED(dev->CreateRenderTargetView(bb.Get(), nullptr,&render_target_view_))) return false; return true; }
void VideoPlayer::renderFrame(ComPtr<ID3D11Texture2D> src){
    if(!src||!render_target_view_) return; 
    ID3D11Device* dev=getD3D11Device(); if(!dev) return; 
    ID3D11DeviceContext* ctx=nullptr; dev->GetImmediateContext(&ctx); if(!ctx) return; 
    if(!fullscreen_vs_||!fullscreen_ps_) if(!initializeRenderPipeline(dev)) return; 
    D3D11_VIEWPORT vp{}; UINT num=1; ctx->RSGetViewports(&num,&vp); 
    switch(render_target_type_){ 
        case RenderTargetType::TEXTURE: 
            if(render_target_texture_){ D3D11_TEXTURE2D_DESC desc; render_target_texture_->GetDesc(&desc); vp.Width=(FLOAT)desc.Width; vp.Height=(FLOAT)desc.Height; }
            break; 
        case RenderTargetType::SWAPCHAIN: 
            if(render_target_swapchain_){ DXGI_SWAP_CHAIN_DESC desc; render_target_swapchain_->GetDesc(&desc); vp.Width=(FLOAT)desc.BufferDesc.Width; vp.Height=(FLOAT)desc.BufferDesc.Height; }
            break; 
    }
    vp.MinDepth=0; vp.MaxDepth=1; ctx->RSSetViewports(1,&vp); 
    ctx->OMSetRenderTargets(1, render_target_view_.GetAddressOf(), nullptr); 
    FLOAT clear[4]={0,0,0,1}; ctx->ClearRenderTargetView(render_target_view_.Get(), clear); 
    // For now just copy the decoded RGB texture to render target if possible (only works for same size/format)
    if(render_target_type_==RenderTargetType::TEXTURE && render_target_texture_){
        ctx->CopyResource(render_target_texture_.Get(), src.Get());
    } else if(render_target_type_==RenderTargetType::SWAPCHAIN){
        // Need back buffer pointer for copy, fetch it each frame (could cache)
        ComPtr<ID3D11Texture2D> backBuffer;
        if(SUCCEEDED(render_target_swapchain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)backBuffer.GetAddressOf()))){
            ctx->CopyResource(backBuffer.Get(), src.Get());
        }
    }
    if(render_target_type_==RenderTargetType::SWAPCHAIN && render_target_swapchain_) render_target_swapchain_->Present(1,0); 
}
bool VideoPlayer::initializeRenderPipeline(ID3D11Device* device){ if(!device) return false; if(fullscreen_vs_ && fullscreen_ps_) return true; // Minimal pipeline not actually drawing geometry (we use CopyResource)
    // Create dummy states
    D3D11_BLEND_DESC bd={}; bd.RenderTarget[0].RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL; if(FAILED(device->CreateBlendState(&bd,&blend_state_))) return false; D3D11_RASTERIZER_DESC rd={}; rd.FillMode=D3D11_FILL_SOLID; rd.CullMode=D3D11_CULL_NONE; rd.DepthClipEnable=TRUE; if(FAILED(device->CreateRasterizerState(&rd,&raster_state_))) return false; return true; }
int VideoPlayer::getWidth() const { if(!isOpen()) return 0; return rgb_decoder_->getWidth(); }
int VideoPlayer::getHeight() const { if(!isOpen()) return 0; return rgb_decoder_->getHeight(); }
