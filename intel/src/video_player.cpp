#include "video_player.h"
#include <iostream>
#include <chrono>
#include <d3dcompiler.h>
#include <fstream>
#include <vector>
#include <comdef.h>
#include <algorithm>

extern "C" {
#include <libavutil/rational.h>
#include <libavutil/frame.h>
}


VideoPlayer::VideoPlayer() 
    : rgb_decoder_(std::make_unique<RgbVideoDecoder>()) {
}


VideoPlayer::~VideoPlayer() {
    close();
}


bool VideoPlayer::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    // 打开 RGB 视频解码器
    if (!rgb_decoder_->open(filepath)) {
        std::cerr << "Failed to open RGB decoder for: " << filepath << std::endl;
        return false;
    }
    
    return true;
}


bool VideoPlayer::setRenderTarget(ComPtr<ID3D11Texture2D> render_target) {
    if (!isOpen()) {
        std::cerr << "Cannot set render target: video not opened" << std::endl;
        return false;
    }
    
    // 设置渲染目标类型
    render_target_type_ = RenderTargetType::TEXTURE;
    render_target_texture_ = render_target;
    
    // 初始化渲染目标
    if (!initializeRenderTarget(render_target)) {
        std::cerr << "Failed to initialize texture render target" << std::endl;
        return false;
    }
    
    // 初始化渲染管线
    if (!initializeRenderPipeline(getD3D11Device())) {
        std::cerr << "Failed to initialize render pipeline" << std::endl;
        return false;
    }
    
    return true;
}


bool VideoPlayer::setRenderTarget(ComPtr<IDXGISwapChain> swap_chain) {
    if (!isOpen()) {
        std::cerr << "Cannot set render target: video not opened" << std::endl;
        return false;
    }
    
    // 设置渲染目标类型
    render_target_type_ = RenderTargetType::SWAPCHAIN;
    render_target_swapchain_ = swap_chain;
    
    // 初始化渲染目标
    if (!initializeRenderTarget(swap_chain)) {
        std::cerr << "Failed to initialize swapchain render target" << std::endl;
        return false;
    }
    
    // 初始化渲染管线
    if (!initializeRenderPipeline(getD3D11Device())) {
        std::cerr << "Failed to initialize render pipeline" << std::endl;
        return false;
    }
    
    return true;
}


void VideoPlayer::onTimer(double current_time) {
    if (!isReady()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // 预加载下一帧到缓冲区
    preloadNextFrame();
    
    // 查找要显示的帧
    bool should_render = false;
    ComPtr<ID3D11Texture2D> texture_to_render;
    
    while (!frame_buffer_.empty()) {
        const CachedFrame& front_frame = frame_buffer_.front();
        
        if (front_frame.frame_time <= current_time + 0.033) { // 33ms 容差
            // 这一帧应该被显示
            current_frame_texture_ = front_frame.rgb_frame.rgb_texture;
            current_frame_time_ = front_frame.frame_time;
            texture_to_render = front_frame.rgb_frame.rgb_texture;
            should_render = true;
            
            // 释放 AVFrame
            if (front_frame.rgb_frame.hw_frame.frame) {
                av_frame_free(const_cast<AVFrame**>(&front_frame.rgb_frame.hw_frame.frame));
            }
            
            frame_buffer_.pop();
        } else {
            break;
        }
    }
    
    // 渲染当前帧
    if (should_render && texture_to_render) {
        renderFrame(texture_to_render);
    }
}


ComPtr<ID3D11Texture2D> VideoPlayer::getCurrentFrame() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return current_frame_texture_;
}


void VideoPlayer::close() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // 清理帧缓冲区
    while (!frame_buffer_.empty()) {
        const CachedFrame& front_frame = frame_buffer_.front();
        if (front_frame.rgb_frame.hw_frame.frame) {
            av_frame_free(const_cast<AVFrame**>(&front_frame.rgb_frame.hw_frame.frame));
        }
        frame_buffer_.pop();
    }
    
    // 重置状态
    current_frame_texture_.Reset();
    current_frame_time_ = 0.0;
    
    // 清理渲染资源
    render_target_texture_.Reset();
    render_target_swapchain_.Reset();
    render_target_view_.Reset();
    fullscreen_vs_.Reset();
    fullscreen_ps_.Reset();
    texture_sampler_.Reset();
    blend_state_.Reset();
    raster_state_.Reset();
    current_texture_srv_.Reset();
    
    // 关闭解码器
    if (rgb_decoder_) {
        rgb_decoder_->close();
    }
}


bool VideoPlayer::isOpen() const {
    return rgb_decoder_ && rgb_decoder_->isOpen();
}


bool VideoPlayer::isReady() const {
    return isOpen() && render_target_view_ != nullptr;
}


bool VideoPlayer::isEOF() const {
    return rgb_decoder_ && rgb_decoder_->isEOF();
}


ID3D11Device* VideoPlayer::getD3D11Device() const {
    if (!rgb_decoder_ || !rgb_decoder_->getHwDecoder()) {
        return nullptr;
    }
    
    return rgb_decoder_->getHwDecoder()->getD3D11Device();
}


ID3D11DeviceContext* VideoPlayer::getD3D11DeviceContext() const {
    if (!rgb_decoder_ || !rgb_decoder_->getHwDecoder()) {
        return nullptr;
    }
    
    return rgb_decoder_->getHwDecoder()->getD3D11DeviceContext();
}


int VideoPlayer::getWidth() const {
    if (!rgb_decoder_) {
        return 0;
    }
    return rgb_decoder_->getWidth();
}


int VideoPlayer::getHeight() const {
    if (!rgb_decoder_) {
        return 0;
    }
    return rgb_decoder_->getHeight();
}


bool VideoPlayer::preloadNextFrame() {
    // 限制缓冲区大小
    if (frame_buffer_.size() >= 5) {
        return true;
    }
    
    RgbVideoDecoder::DecodedRgbFrame rgb_frame;
    if (!rgb_decoder_->readNextFrame(rgb_frame)) {
        return false;
    }
    
    if (!rgb_frame.is_valid || !rgb_frame.hw_frame.frame) {
        return false;
    }
    
    // 计算时间戳
    AVFrame* frame = rgb_frame.hw_frame.frame;
    double frame_time = 0.0;
    
    if (frame->pts != AV_NOPTS_VALUE) {
        // 获取时间基
        AVRational time_base = {1, 25}; // 默认 25fps
        if (rgb_decoder_->getHwDecoder() && rgb_decoder_->getHwDecoder()->getStreamReader()) {
            MKVStreamReader::StreamInfo stream_info = rgb_decoder_->getHwDecoder()->getStreamReader()->getStreamInfo();
            if (stream_info.video_stream_index >= 0) {
                time_base = stream_info.video_time_base;
            }
        }
        frame_time = av_q2d(time_base) * frame->pts;
    }
    
    // 添加到缓冲区
    CachedFrame cached_frame;
    cached_frame.rgb_frame = rgb_frame;
    cached_frame.frame_time = frame_time;
    frame_buffer_.push(cached_frame);
    
    return true;
}


bool VideoPlayer::initializeRenderTarget(ComPtr<ID3D11Texture2D> render_target) {
    ID3D11Device* device = getD3D11Device();
    if (!device) {
        std::cerr << "No D3D11 device available" << std::endl;
        return false;
    }
    
    // 创建 Render Target View
    D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
    rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    rtv_desc.Texture2D.MipSlice = 0;
    
    HRESULT hr = device->CreateRenderTargetView(render_target.Get(), &rtv_desc, &render_target_view_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target view: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}


bool VideoPlayer::initializeRenderTarget(ComPtr<IDXGISwapChain> swap_chain) {
    ID3D11Device* device = getD3D11Device();
    if (!device) {
        std::cerr << "No D3D11 device available" << std::endl;
        return false;
    }
    
    // 获取交换链的后缓冲区
    ComPtr<ID3D11Texture2D> back_buffer;
    HRESULT hr = swap_chain->GetBuffer(0, IID_PPV_ARGS(&back_buffer));
    if (FAILED(hr)) {
        std::cerr << "Failed to get back buffer: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建 Render Target View
    hr = device->CreateRenderTargetView(back_buffer.Get(), nullptr, &render_target_view_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target view: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}


bool VideoPlayer::initializeRenderPipeline(ID3D11Device* device) {
    if (!device) {
        return false;
    }
    
    // 读取着色器文件
    std::ifstream file("src/fullscreen_quad.hlsl", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open shader file" << std::endl;
        return false;
    }
    
    std::vector<char> shader_source((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
    file.close();
    
    // 编译 Vertex Shader
    ComPtr<ID3DBlob> vs_blob, error_blob;
    HRESULT hr = D3DCompile(shader_source.data(), shader_source.size(),
                           "fullscreen_quad.hlsl", nullptr, nullptr,
                           "VSMain", "vs_5_0", 0, 0, &vs_blob, &error_blob);
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Vertex shader compile error: " 
                     << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    
    hr = device->CreateVertexShader(vs_blob->GetBufferPointer(),
                                   vs_blob->GetBufferSize(),
                                   nullptr, &fullscreen_vs_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create vertex shader: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 编译 Pixel Shader
    ComPtr<ID3DBlob> ps_blob;
    hr = D3DCompile(shader_source.data(), shader_source.size(),
                   "fullscreen_quad.hlsl", nullptr, nullptr,
                   "PSMain", "ps_5_0", 0, 0, &ps_blob, &error_blob);
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Pixel shader compile error: " 
                     << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    
    hr = device->CreatePixelShader(ps_blob->GetBufferPointer(),
                                  ps_blob->GetBufferSize(),
                                  nullptr, &fullscreen_ps_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create pixel shader: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建采样器状态
    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    
    hr = device->CreateSamplerState(&sampler_desc, &texture_sampler_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create sampler state: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建混合状态 (禁用混合)
    D3D11_BLEND_DESC blend_desc = {};
    blend_desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    hr = device->CreateBlendState(&blend_desc, &blend_state_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create blend state: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建光栅化状态
    D3D11_RASTERIZER_DESC raster_desc = {};
    raster_desc.CullMode = D3D11_CULL_BACK;
    raster_desc.FillMode = D3D11_FILL_SOLID;
    
    hr = device->CreateRasterizerState(&raster_desc, &raster_state_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create rasterizer state: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}


ComPtr<ID3D11ShaderResourceView> VideoPlayer::createTextureShaderResourceView(ComPtr<ID3D11Texture2D> texture) {
    ID3D11Device* device = getD3D11Device();
    if (!device || !texture) {
        return nullptr;
    }
    
    ComPtr<ID3D11ShaderResourceView> srv;
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.MostDetailedMip = 0;
    
    HRESULT hr = device->CreateShaderResourceView(texture.Get(), &srv_desc, &srv);
    if (FAILED(hr)) {
        std::cerr << "Failed to create shader resource view: 0x" << std::hex << hr << std::endl;
        return nullptr;
    }
    
    return srv;
}


void VideoPlayer::renderFrame(ComPtr<ID3D11Texture2D> source_texture) {
    ID3D11DeviceContext* context = getD3D11DeviceContext();
    if (!context || !source_texture || !render_target_view_) {
        return;
    }
    
    // 创建 SRV
    ComPtr<ID3D11ShaderResourceView> texture_srv = createTextureShaderResourceView(source_texture);
    if (!texture_srv) {
        return;
    }
    
    // 获取渲染目标尺寸
    D3D11_TEXTURE2D_DESC desc;
    if (render_target_type_ == RenderTargetType::TEXTURE && render_target_texture_) {
        render_target_texture_->GetDesc(&desc);
    } else if (render_target_type_ == RenderTargetType::SWAPCHAIN && render_target_swapchain_) {
        ComPtr<ID3D11Texture2D> back_buffer;
        render_target_swapchain_->GetBuffer(0, IID_PPV_ARGS(&back_buffer));
        if (back_buffer) {
            back_buffer->GetDesc(&desc);
        }
    } else {
        return;
    }
    
    // 设置视口
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(desc.Width);
    viewport.Height = static_cast<float>(desc.Height);
    viewport.MaxDepth = 1.0f;
    context->RSSetViewports(1, &viewport);
    
    // 设置渲染目标
    context->OMSetRenderTargets(1, render_target_view_.GetAddressOf(), nullptr);
    
    // 设置着色器
    context->VSSetShader(fullscreen_vs_.Get(), nullptr, 0);
    context->PSSetShader(fullscreen_ps_.Get(), nullptr, 0);
    
    // 设置纹理和采样器
    context->PSSetShaderResources(0, 1, texture_srv.GetAddressOf());
    context->PSSetSamplers(0, 1, texture_sampler_.GetAddressOf());
    
    // 设置渲染状态
    context->OMSetBlendState(blend_state_.Get(), nullptr, 0xffffffff);
    context->RSSetState(raster_state_.Get());
    
    // 渲染全屏三角形 (无需顶点缓冲区)
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->Draw(3, 0);
    
    // 如果是交换链，则呈现
    if (render_target_type_ == RenderTargetType::SWAPCHAIN && render_target_swapchain_) {
        render_target_swapchain_->Present(1, 0);
    }
}