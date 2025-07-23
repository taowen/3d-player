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
    : stereo_decoder_(std::make_unique<StereoVideoDecoder>()) {
}


VideoPlayer::~VideoPlayer() {
    close();
}


bool VideoPlayer::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    // 打开立体解码器
    if (!stereo_decoder_->open(filepath)) {
        std::cerr << "Failed to open stereo decoder for: " << filepath << std::endl;
        return false;
    }
    
    // StereoVideoDecoder 不暴露内部实现细节，无需获取底层组件
    // 时间戳信息从 DecodedStereoFrame 的 AVFrame 中计算获取
    
    return true;
}


bool VideoPlayer::setRenderTarget(ComPtr<ID3D11Texture2D> render_target) {
    if (!isOpen()) {
        std::cerr << "Cannot set render target: video not opened" << std::endl;
        return false;
    }
    
    // 清理之前的渲染目标
    render_target_view_.Reset();
    render_target_texture_.Reset();
    render_target_swapchain_.Reset();
    
    // 初始化渲染目标（纹理）
    if (!initializeRenderTarget(render_target)) {
        std::cerr << "Failed to initialize render target texture" << std::endl;
        return false;
    }
    
    // 预解码第一帧
    if (!preloadNextFrame()) {
        std::cerr << "Failed to preload first frame" << std::endl;
        return false;
    }
    
    return true;
}


bool VideoPlayer::setRenderTarget(ComPtr<IDXGISwapChain> swap_chain) {
    if (!isOpen()) {
        std::cerr << "Cannot set render target: video not opened" << std::endl;
        return false;
    }
    
    // 清理之前的渲染目标
    render_target_view_.Reset();
    render_target_texture_.Reset();
    render_target_swapchain_.Reset();
    
    // 初始化渲染目标（交换链）
    if (!initializeRenderTarget(swap_chain)) {
        std::cerr << "Failed to initialize render target swapchain" << std::endl;
        return false;
    }
    
    // 预解码第一帧
    if (!preloadNextFrame()) {
        std::cerr << "Failed to preload first frame" << std::endl;
        return false;
    }
    
    return true;
}


void VideoPlayer::onTimer(double current_time) {
    if (!isReady()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // 检查是否需要切换到下一帧
    bool need_render = false;
    
    while (!frame_buffer_.empty()) {
        auto& front_frame = frame_buffer_.front();
        // 从AVFrame和时间基计算PTS时间戳
        double frame_time = 0.0;
        if (front_frame.frame && front_frame.frame->pts != AV_NOPTS_VALUE) {
            AVRational time_base = stereo_decoder_->getVideoTimeBase();
            frame_time = front_frame.frame->pts * av_q2d(time_base);
        }
        
        if (frame_time <= current_time) {
            // 时间已到，切换到这一帧
            if (current_frame_texture_) {
                // 释放当前帧
                current_frame_texture_.Reset();
            }
            
            // 设置新的当前帧
            current_frame_texture_ = front_frame.stereo_texture;
            current_frame_time_ = frame_time;
            
            // 释放AVFrame内存
            if (front_frame.frame) {
                av_frame_free(&front_frame.frame);
            }
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
        DecodedStereoFrame& frame = frame_buffer_.front();
        if (frame.frame) {
            av_frame_free(&frame.frame);
        }
        frame_buffer_.pop();
    }
    
    // 释放当前帧
    current_frame_texture_.Reset();
    current_frame_time_ = 0.0;
    
    // 释放渲染目标相关资源
    render_target_view_.Reset();
    render_target_texture_.Reset();
    render_target_swapchain_.Reset();
    
    // 释放渲染管线资源
    fullscreen_vs_.Reset();
    fullscreen_ps_.Reset();
    texture_sampler_.Reset();
    blend_state_.Reset();
    raster_state_.Reset();
    current_texture_srv_.Reset();
    
    // 关闭解码器
    if (stereo_decoder_) {
        stereo_decoder_->close();
    }
}


bool VideoPlayer::isOpen() const {
    return stereo_decoder_ && stereo_decoder_->isOpen();
}


bool VideoPlayer::isReady() const {
    return isOpen() && render_target_view_;
}


bool VideoPlayer::isEOF() const {
    return stereo_decoder_ && stereo_decoder_->isEOF();
}


StereoVideoDecoder* VideoPlayer::getStereoDecoder() const {
    return stereo_decoder_.get();
}


ID3D11Device* VideoPlayer::getD3D11Device() const {
    if (!isOpen()) {
        return nullptr;
    }
    
    return stereo_decoder_->getD3D11Device();
}


ID3D11DeviceContext* VideoPlayer::getD3D11DeviceContext() const {
    if (!isOpen()) {
        return nullptr;
    }
    
    // StereoVideoDecoder 不暴露 DeviceContext，通过设备获取
    ID3D11Device* device = stereo_decoder_->getD3D11Device();
    if (!device) {
        return nullptr;
    }
    
    ID3D11DeviceContext* context = nullptr;
    device->GetImmediateContext(&context);
    return context;
}


bool VideoPlayer::preloadNextFrame() {
    if (!stereo_decoder_) {
        return false;
    }
    
    // 限制缓冲区大小，避免内存占用过多
    if (frame_buffer_.size() >= 3) {
        return true;  // 缓冲区已满，不需要预加载更多帧
    }
    
    DecodedStereoFrame frame;
    if (!stereo_decoder_->readNextFrame(frame)) {
        return false;  // 读取失败或到达文件末尾
    }
    
    // 将帧加入缓冲队列
    frame_buffer_.push(frame);
    
    return true;
}


// convertPtsToSeconds 函数已移除，直接从 AVFrame 计算时间戳


bool VideoPlayer::initializeRenderTarget(ComPtr<ID3D11Texture2D> render_target) {
    if (!render_target) {
        std::cerr << "Invalid render target texture" << std::endl;
        return false;
    }
    
    // 保存渲染目标信息
    render_target_type_ = RenderTargetType::TEXTURE;
    render_target_texture_ = render_target;
    
    // 获取 D3D11 设备
    ID3D11Device* device = stereo_decoder_->getD3D11Device();
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
    ID3D11Device* device = stereo_decoder_->getD3D11Device();
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
    ID3D11Device* device = stereo_decoder_->getD3D11Device();
    if (!device) {
        return;
    }
    
    ID3D11DeviceContext* context = nullptr;
    device->GetImmediateContext(&context);
    if (!context) {
        return;
    }
    
    // 确保渲染管线已初始化
    if (!fullscreen_vs_ || !fullscreen_ps_) {
        if (!initializeRenderPipeline(device)) {
            std::cerr << "Failed to initialize render pipeline" << std::endl;
            return;
        }
    }
    
    // 为源纹理创建 SRV
    ComPtr<ID3D11ShaderResourceView> source_srv = createTextureShaderResourceView(source_texture);
    if (!source_srv) {
        return;
    }
    
    // 设置视口
    D3D11_VIEWPORT viewport = {};
    UINT num_viewports = 1;
    context->RSGetViewports(&num_viewports, &viewport);
    
    // 根据渲染目标类型设置视口大小
    switch (render_target_type_) {
        case RenderTargetType::TEXTURE:
            if (render_target_texture_) {
                D3D11_TEXTURE2D_DESC desc;
                render_target_texture_->GetDesc(&desc);
                viewport.Width = (FLOAT)desc.Width;
                viewport.Height = (FLOAT)desc.Height;
                viewport.MinDepth = 0.0f;
                viewport.MaxDepth = 1.0f;
                viewport.TopLeftX = 0.0f;
                viewport.TopLeftY = 0.0f;
            }
            break;
        case RenderTargetType::SWAPCHAIN:
            if (render_target_swapchain_) {
                DXGI_SWAP_CHAIN_DESC desc;
                render_target_swapchain_->GetDesc(&desc);
                viewport.Width = (FLOAT)desc.BufferDesc.Width;
                viewport.Height = (FLOAT)desc.BufferDesc.Height;
                viewport.MinDepth = 0.0f;
                viewport.MaxDepth = 1.0f;
                viewport.TopLeftX = 0.0f;
                viewport.TopLeftY = 0.0f;
            }
            break;
    }
    
    // 设置渲染管线状态
    context->RSSetViewports(1, &viewport);
    context->OMSetRenderTargets(1, render_target_view_.GetAddressOf(), nullptr);
    
    // 清屏操作，确保没有残留内容
    FLOAT clear_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; // 黑色背景
    context->ClearRenderTargetView(render_target_view_.Get(), clear_color);
    
    context->RSSetState(raster_state_.Get());
    context->OMSetBlendState(blend_state_.Get(), nullptr, 0xFFFFFFFF);
    
    // 设置 shaders
    context->VSSetShader(fullscreen_vs_.Get(), nullptr, 0);
    context->PSSetShader(fullscreen_ps_.Get(), nullptr, 0);
    
    // 设置纹理和采样器
    context->PSSetShaderResources(0, 1, source_srv.GetAddressOf());
    context->PSSetSamplers(0, 1, texture_sampler_.GetAddressOf());
    
    // 设置输入装配器
    context->IASetInputLayout(nullptr);  // 使用无顶点缓冲区技术
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // 渲染全屏三角形
    context->Draw(3, 0);
    
    // 对于交换链，需要 Present
    if (render_target_type_ == RenderTargetType::SWAPCHAIN && render_target_swapchain_) {
        render_target_swapchain_->Present(1, 0);
    }
}


bool VideoPlayer::initializeRenderPipeline(ID3D11Device* device) {
    if (!device) {
        return false;
    }
    
    // 编译 shader 文件
    ComPtr<ID3DBlob> vs_blob;
    ComPtr<ID3DBlob> ps_blob;
    ComPtr<ID3DBlob> error_blob;
    
    // 编译 vertex shader
    HRESULT hr = D3DCompileFromFile(
        L"src/fullscreen_quad.hlsl",
        nullptr,
        nullptr,
        "VSMain",
        "vs_5_0",
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
        0,
        &vs_blob,
        &error_blob
    );
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Vertex shader compilation error: " << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    
    // 编译 pixel shader
    hr = D3DCompileFromFile(
        L"src/fullscreen_quad.hlsl",
        nullptr,
        nullptr,
        "PSMain",
        "ps_5_0",
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
        0,
        &ps_blob,
        &error_blob
    );
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Pixel shader compilation error: " << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    
    // 创建 vertex shader
    hr = device->CreateVertexShader(
        vs_blob->GetBufferPointer(),
        vs_blob->GetBufferSize(),
        nullptr,
        &fullscreen_vs_
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create vertex shader" << std::endl;
        return false;
    }
    
    // 创建 pixel shader
    hr = device->CreatePixelShader(
        ps_blob->GetBufferPointer(),
        ps_blob->GetBufferSize(),
        nullptr,
        &fullscreen_ps_
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create pixel shader" << std::endl;
        return false;
    }
    
    // 创建采样器状态
    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0.0f;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    
    hr = device->CreateSamplerState(&sampler_desc, &texture_sampler_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create sampler state" << std::endl;
        return false;
    }
    
    // 创建混合状态
    D3D11_BLEND_DESC blend_desc = {};
    blend_desc.AlphaToCoverageEnable = FALSE;
    blend_desc.IndependentBlendEnable = FALSE;
    blend_desc.RenderTarget[0].BlendEnable = FALSE;
    blend_desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    blend_desc.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
    blend_desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blend_desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blend_desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blend_desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blend_desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    hr = device->CreateBlendState(&blend_desc, &blend_state_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create blend state" << std::endl;
        return false;
    }
    
    // 创建光栅化状态
    D3D11_RASTERIZER_DESC raster_desc = {};
    raster_desc.FillMode = D3D11_FILL_SOLID;
    raster_desc.CullMode = D3D11_CULL_NONE;
    raster_desc.FrontCounterClockwise = FALSE;
    raster_desc.DepthBias = 0;
    raster_desc.DepthBiasClamp = 0.0f;
    raster_desc.SlopeScaledDepthBias = 0.0f;
    raster_desc.DepthClipEnable = TRUE;
    raster_desc.ScissorEnable = FALSE;
    raster_desc.MultisampleEnable = FALSE;
    raster_desc.AntialiasedLineEnable = FALSE;
    
    hr = device->CreateRasterizerState(&raster_desc, &raster_state_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create rasterizer state" << std::endl;
        return false;
    }
    
    return true;
}


ComPtr<ID3D11ShaderResourceView> VideoPlayer::createTextureShaderResourceView(ComPtr<ID3D11Texture2D> texture) {
    if (!texture) {
        return nullptr;
    }
    
    // 获取设备
    ID3D11Device* device = stereo_decoder_->getD3D11Device();
    if (!device) {
        return nullptr;
    }
    
    // 获取纹理描述
    D3D11_TEXTURE2D_DESC texture_desc;
    texture->GetDesc(&texture_desc);
    
    // 创建 SRV 描述
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = texture_desc.Format;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = texture_desc.MipLevels;
    
    // 创建 SRV
    ComPtr<ID3D11ShaderResourceView> srv;
    HRESULT hr = device->CreateShaderResourceView(texture.Get(), &srv_desc, &srv);
    if (FAILED(hr)) {
        std::cerr << "Failed to create shader resource view" << std::endl;
        return nullptr;
    }
    
    return srv;
}

int VideoPlayer::getWidth() const {
    if (!isOpen()) {
        return 0;
    }
    
    return stereo_decoder_->getWidth();
}

int VideoPlayer::getHeight() const {
    if (!isOpen()) {
        return 0;
    }
    
    return stereo_decoder_->getHeight();
} 