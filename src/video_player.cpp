#include "video_player.h"
#include <iostream>
#include <chrono>

extern "C" {
#include <libavutil/rational.h>
}


VideoPlayer::VideoPlayer() 
    : rgb_decoder_(std::make_unique<RgbVideoDecoder>()) {
}


VideoPlayer::~VideoPlayer() {
    close();
}


bool VideoPlayer::open(const std::string& filepath, ComPtr<ID3D11Texture2D> render_target) {
    if (isOpen()) {
        close();
    }
    
    // 打开 RGB 解码器
    if (!rgb_decoder_->open(filepath)) {
        std::cerr << "Failed to open RGB decoder for: " << filepath << std::endl;
        return false;
    }
    
    // 获取视频信息
    HwVideoDecoder* hw_decoder = rgb_decoder_->getHwDecoder();
    if (!hw_decoder) {
        std::cerr << "Failed to get hardware decoder" << std::endl;
        close();
        return false;
    }
    
    MKVStreamReader* stream_reader = hw_decoder->getStreamReader();
    if (!stream_reader) {
        std::cerr << "Failed to get stream reader" << std::endl;
        close();
        return false;
    }
    
    // 获取流信息以计算时间戳
    stream_info_ = stream_reader->getStreamInfo();
    if (stream_info_.video_stream_index < 0) {
        std::cerr << "No video stream found" << std::endl;
        close();
        return false;
    }
    
    // 初始化渲染目标（纹理）
    if (!initializeRenderTarget(render_target)) {
        std::cerr << "Failed to initialize render target texture" << std::endl;
        close();
        return false;
    }
    
    // 预解码第一帧
    if (!preloadNextFrame()) {
        std::cerr << "Failed to preload first frame" << std::endl;
        close();
        return false;
    }
    
    return true;
}


bool VideoPlayer::open(const std::string& filepath, ComPtr<IDXGISwapChain> swap_chain) {
    if (isOpen()) {
        close();
    }
    
    // 打开 RGB 解码器
    if (!rgb_decoder_->open(filepath)) {
        std::cerr << "Failed to open RGB decoder for: " << filepath << std::endl;
        return false;
    }
    
    // 获取视频信息
    HwVideoDecoder* hw_decoder = rgb_decoder_->getHwDecoder();
    if (!hw_decoder) {
        std::cerr << "Failed to get hardware decoder" << std::endl;
        close();
        return false;
    }
    
    MKVStreamReader* stream_reader = hw_decoder->getStreamReader();
    if (!stream_reader) {
        std::cerr << "Failed to get stream reader" << std::endl;
        close();
        return false;
    }
    
    // 获取流信息以计算时间戳
    stream_info_ = stream_reader->getStreamInfo();
    if (stream_info_.video_stream_index < 0) {
        std::cerr << "No video stream found" << std::endl;
        close();
        return false;
    }
    
    // 初始化渲染目标（交换链）
    if (!initializeRenderTarget(swap_chain)) {
        std::cerr << "Failed to initialize render target swapchain" << std::endl;
        close();
        return false;
    }
    
    // 预解码第一帧
    if (!preloadNextFrame()) {
        std::cerr << "Failed to preload first frame" << std::endl;
        close();
        return false;
    }
    
    return true;
}


void VideoPlayer::onTimer(double current_time) {
    if (!isOpen()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // 检查是否需要切换到下一帧
    bool need_render = false;
    
    while (!frame_buffer_.empty()) {
        auto& front_frame = frame_buffer_.front();
        double frame_time = convertPtsToSeconds(front_frame.hw_frame.frame);
        
        if (frame_time <= current_time) {
            // 时间已到，切换到这一帧
            if (current_frame_texture_) {
                // 释放当前帧
                current_frame_texture_.Reset();
            }
            
            // 设置新的当前帧
            current_frame_texture_ = front_frame.rgb_texture;
            current_frame_time_ = frame_time;
            
            // 释放硬件帧内存
            av_frame_free(&front_frame.hw_frame.frame);
            frame_buffer_.pop();
            
            need_render = true;
            
            // 预解码下一帧
            preloadNextFrame();
        } else {
            // 时间还没到，停止切换
            break;
        }
    }
    
    // 如果需要渲染，则内部调用渲染
    if (need_render && current_frame_texture_) {
        renderFrame(current_frame_texture_);
    }
}


ComPtr<ID3D11Texture2D> VideoPlayer::getCurrentFrame() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return current_frame_texture_;
}


void VideoPlayer::close() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // 清空帧缓冲
    while (!frame_buffer_.empty()) {
        auto& frame = frame_buffer_.front();
        av_frame_free(&frame.hw_frame.frame);
        frame_buffer_.pop();
    }
    
    // 释放当前帧
    current_frame_texture_.Reset();
    current_frame_time_ = 0.0;
    
    // 释放渲染目标相关资源
    render_target_view_.Reset();
    render_target_texture_.Reset();
    render_target_swapchain_.Reset();
    
    // 关闭解码器
    if (rgb_decoder_) {
        rgb_decoder_->close();
    }
}


bool VideoPlayer::isOpen() const {
    return rgb_decoder_ && rgb_decoder_->isOpen() && render_target_view_;
}


bool VideoPlayer::isEOF() const {
    return rgb_decoder_ && rgb_decoder_->isEOF();
}


RgbVideoDecoder* VideoPlayer::getRgbDecoder() const {
    return rgb_decoder_.get();
}


bool VideoPlayer::preloadNextFrame() {
    if (!rgb_decoder_) {
        return false;
    }
    
    // 限制缓冲区大小，避免内存占用过多
    if (frame_buffer_.size() >= 3) {
        return true;  // 缓冲区已满，不需要预加载更多帧
    }
    
    RgbVideoDecoder::DecodedRgbFrame frame;
    if (!rgb_decoder_->readNextFrame(frame)) {
        return false;  // 读取失败或到达文件末尾
    }
    
    // 将帧加入缓冲队列
    frame_buffer_.push(frame);
    
    return true;
}


double VideoPlayer::convertPtsToSeconds(AVFrame* frame) const {
    if (!frame || frame->pts == AV_NOPTS_VALUE) {
        return 0.0;
    }
    
    // 将 PTS 转换为秒数：pts * time_base.num / time_base.den
    return (double)frame->pts * stream_info_.video_time_base.num / stream_info_.video_time_base.den;
}


bool VideoPlayer::initializeRenderTarget(ComPtr<ID3D11Texture2D> render_target) {
    if (!render_target) {
        std::cerr << "Invalid render target texture" << std::endl;
        return false;
    }
    
    // 保存渲染目标信息
    render_target_type_ = RenderTargetType::TEXTURE;
    render_target_texture_ = render_target;
    
    // 获取 D3D11 设备
    HwVideoDecoder* hw_decoder = rgb_decoder_->getHwDecoder();
    if (!hw_decoder) {
        std::cerr << "Failed to get hardware decoder" << std::endl;
        return false;
    }
    
    ID3D11Device* device = hw_decoder->getD3D11Device();
    if (!device) {
        std::cerr << "Failed to get D3D11 device" << std::endl;
        return false;
    }
    
    // 创建渲染目标视图
    D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
    rtv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    rtv_desc.Texture2D.MipSlice = 0;
    
    HRESULT hr = device->CreateRenderTargetView(render_target.Get(), &rtv_desc, &render_target_view_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target view" << std::endl;
        return false;
    }
    
    return true;
}


bool VideoPlayer::initializeRenderTarget(ComPtr<IDXGISwapChain> swap_chain) {
    if (!swap_chain) {
        std::cerr << "Invalid swap chain" << std::endl;
        return false;
    }
    
    // 保存渲染目标信息
    render_target_type_ = RenderTargetType::SWAPCHAIN;
    render_target_swapchain_ = swap_chain;
    
    // 获取 D3D11 设备
    HwVideoDecoder* hw_decoder = rgb_decoder_->getHwDecoder();
    if (!hw_decoder) {
        std::cerr << "Failed to get hardware decoder" << std::endl;
        return false;
    }
    
    ID3D11Device* device = hw_decoder->getD3D11Device();
    if (!device) {
        std::cerr << "Failed to get D3D11 device" << std::endl;
        return false;
    }
    
    // 从交换链获取后台缓冲区
    ComPtr<ID3D11Texture2D> back_buffer;
    HRESULT hr = swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer);
    if (FAILED(hr)) {
        std::cerr << "Failed to get back buffer from swap chain" << std::endl;
        return false;
    }
    
    // 创建渲染目标视图
    hr = device->CreateRenderTargetView(back_buffer.Get(), nullptr, &render_target_view_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target view for swap chain" << std::endl;
        return false;
    }
    
    return true;
}


void VideoPlayer::renderFrame(ComPtr<ID3D11Texture2D> source_texture) {
    if (!source_texture || !render_target_view_) {
        return;
    }
    
    // 获取 D3D11 设备上下文
    HwVideoDecoder* hw_decoder = rgb_decoder_->getHwDecoder();
    if (!hw_decoder) {
        return;
    }
    
    ID3D11DeviceContext* context = hw_decoder->getD3D11DeviceContext();
    if (!context) {
        return;
    }
    
    // 根据渲染目标类型选择不同的渲染方式
    switch (render_target_type_) {
        case RenderTargetType::TEXTURE:
            // 渲染到纹理：直接复制纹理
            if (render_target_texture_) {
                context->CopyResource(render_target_texture_.Get(), source_texture.Get());
            }
            break;
            
        case RenderTargetType::SWAPCHAIN:
            // 渲染到交换链：使用渲染管线
            // 简单实现：这里可以扩展为更复杂的渲染管线
            if (render_target_swapchain_) {
                // 获取当前后台缓冲区
                ComPtr<ID3D11Texture2D> back_buffer;
                HRESULT hr = render_target_swapchain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer);
                if (SUCCEEDED(hr)) {
                    context->CopyResource(back_buffer.Get(), source_texture.Get());
                    // 呈现到屏幕
                    render_target_swapchain_->Present(1, 0);
                }
            }
            break;
    }
} 