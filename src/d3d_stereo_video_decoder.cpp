// ⚠️⚠️⚠️ 绝对禁止 CUDA kernel！这是纯 D3D11 compute shader 实现！！！ ⚠️⚠️⚠️
// 本文件严格使用 D3D11 Compute Shader 进行 GPU 计算，禁止任何 CUDA kernel！
// 只允许必要的 CUDA 内存拷贝用于数据传输，绝不允许 CUDA kernel 计算！

#include "d3d_stereo_video_decoder.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>  // 仅用于数据传输，绝不用于 kernel 计算！


D3dStereoVideoDecoder::D3dStereoVideoDecoder() 
    : stereo_decoder_(std::make_unique<StereoVideoDecoder>())
    , d3d_initialized_(false)
    , width_(0)
    , height_(0)
{
}

D3dStereoVideoDecoder::~D3dStereoVideoDecoder() {
    close();
}

bool D3dStereoVideoDecoder::open(const std::string& filepath) {
    if (!stereo_decoder_->open(filepath)) {
        return false;
    }
    
    width_ = stereo_decoder_->getWidth();
    height_ = stereo_decoder_->getHeight();
    
    return initializeD3DComputeShader();
}

void D3dStereoVideoDecoder::close() {
    cleanupResources();
    if (stereo_decoder_) {
        stereo_decoder_->close();
    }
}

bool D3dStereoVideoDecoder::isOpen() const {
    return stereo_decoder_ && stereo_decoder_->isOpen();
}

bool D3dStereoVideoDecoder::isEOF() const {
    return stereo_decoder_ && stereo_decoder_->isEOF();
}

bool D3dStereoVideoDecoder::readNextFrame(DecodedStereoFrameD3D& frame) {
    // 清理之前的状态
    frame.d3d_conversion_valid = false;
    frame.d3d_texture.Reset();
    
    // 读取立体帧
    if (!stereo_decoder_->readNextFrame(frame.stereo_frame)) {
        return false;
    }
    
    // 转换 CUDA 输出到 D3D11 纹理
    if (convertCudaToD3DTexture(frame.stereo_frame)) {
        frame.d3d_texture = d3d_texture_;
        frame.d3d_conversion_valid = true;
    }
    
    return true;
}

int D3dStereoVideoDecoder::getWidth() const {
    return width_;
}

int D3dStereoVideoDecoder::getHeight() const {
    return height_;
}

ID3D11Device* D3dStereoVideoDecoder::getD3D11Device() const {
    return stereo_decoder_ ? stereo_decoder_->getD3D11Device() : nullptr;
}

AVRational D3dStereoVideoDecoder::getVideoTimeBase() const {
    return stereo_decoder_ ? stereo_decoder_->getVideoTimeBase() : AVRational{0, 1};
}

bool D3dStereoVideoDecoder::initializeD3DComputeShader() {
    if (d3d_initialized_) {
        return true;
    }
    
    if (width_ <= 0 || height_ <= 0) {
        std::cerr << "Invalid video dimensions for D3D11 compute shader: " << width_ << "x" << height_ << std::endl;
        return false;
    }
    
    ID3D11Device* d3d_device = getD3D11Device();
    if (!d3d_device) {
        std::cerr << "No D3D11 device available" << std::endl;
        return false;
    }
    
    // 创建并编译 compute shader
    std::ifstream shader_file("shaders/bchw_to_rgba.hlsl");
    if (!shader_file.is_open()) {
        std::cerr << "Failed to open shader file: shaders/bchw_to_rgba.hlsl" << std::endl;
        return false;
    }
    
    std::string shader_source((std::istreambuf_iterator<char>(shader_file)),
                              std::istreambuf_iterator<char>());
    shader_file.close();
    
    ComPtr<ID3DBlob> shader_blob;
    ComPtr<ID3DBlob> error_blob;
    HRESULT hr = D3DCompile(
        shader_source.c_str(),
        shader_source.length(),
        "bchw_to_rgba.hlsl",
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        0,
        0,
        &shader_blob,
        &error_blob
    );
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Shader compilation error: " << (char*)error_blob->GetBufferPointer() << std::endl;
        } else {
            std::cerr << "Failed to compile compute shader: HRESULT = 0x" << std::hex << hr << std::endl;
        }
        return false;
    }
    
    hr = d3d_device->CreateComputeShader(
        shader_blob->GetBufferPointer(),
        shader_blob->GetBufferSize(),
        nullptr,
        &compute_shader_
    );
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader: HRESULT = 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建支持 CUDA 互操作的输入缓冲区（存储 BCHW 平面数据）
    size_t total_floats = width_ * height_ * 4; // BCHW 4 个通道
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.ByteWidth = total_floats * sizeof(float);
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;  // 改为 DEFAULT 以支持 CUDA 互操作
    buffer_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    buffer_desc.CPUAccessFlags = 0;  // 不需要 CPU 访问
    buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;  // 用于 CUDA 互操作
    buffer_desc.StructureByteStride = 0;  // Raw buffer 不需要 stride
    
    hr = d3d_device->CreateBuffer(&buffer_desc, nullptr, &input_buffer_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create input buffer: HRESULT = 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 注册 D3D11 缓冲区到 CUDA 互操作
    cudaError_t cuda_err = cudaGraphicsD3D11RegisterResource(
        reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 
        input_buffer_.Get(), 
        cudaGraphicsRegisterFlagsNone
    );
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to register CUDA graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    // 创建输入 SRV（使用 raw buffer 格式）
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    srv_desc.BufferEx.FirstElement = 0;
    srv_desc.BufferEx.NumElements = total_floats;
    srv_desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    
    hr = d3d_device->CreateShaderResourceView(input_buffer_.Get(), &srv_desc, &input_srv_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create input SRV: HRESULT = 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建输出纹理
    D3D11_TEXTURE2D_DESC texture_desc = {};
    texture_desc.Width = width_;
    texture_desc.Height = height_;
    texture_desc.MipLevels = 1;
    texture_desc.ArraySize = 1;
    texture_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.Usage = D3D11_USAGE_DEFAULT;
    texture_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    texture_desc.CPUAccessFlags = 0;
    texture_desc.MiscFlags = 0;
    
    hr = d3d_device->CreateTexture2D(&texture_desc, nullptr, &d3d_texture_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output texture: HRESULT = 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建输出 UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uav_desc.Texture2D.MipSlice = 0;
    
    hr = d3d_device->CreateUnorderedAccessView(d3d_texture_.Get(), &uav_desc, &output_uav_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output UAV: HRESULT = 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建常量缓冲区
    struct ShaderConstants {
        UINT ImageWidth;
        UINT ImageHeight;
        UINT Padding[2];
    };
    
    D3D11_BUFFER_DESC const_desc = {};
    const_desc.ByteWidth = sizeof(ShaderConstants);
    const_desc.Usage = D3D11_USAGE_DYNAMIC;
    const_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    const_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    
    hr = d3d_device->CreateBuffer(&const_desc, nullptr, &constant_buffer_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create constant buffer: HRESULT = 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    d3d_initialized_ = true;
    std::cout << "D3D11 compute shader initialized successfully: " << width_ << "x" << height_ << " RGBA32F" << std::endl;
    
    return true;
}

bool D3dStereoVideoDecoder::convertCudaToD3DTexture(const DecodedStereoFrame& stereo_frame) {
    if (!d3d_initialized_) {
        std::cerr << "D3D11 compute shader not initialized" << std::endl;
        return false;
    }
    
    if (!stereo_frame.is_valid || !stereo_frame.cuda_output_buffer) {
        std::cerr << "Invalid stereo frame for D3D conversion" << std::endl;
        return false;
    }
    
    ID3D11Device* d3d_device = getD3D11Device();
    ID3D11DeviceContext* d3d_context = nullptr;
    d3d_device->GetImmediateContext(&d3d_context);
    
    if (!d3d_context) {
        std::cerr << "Failed to get D3D11 device context" << std::endl;
        return false;
    }
    
    // 映射 CUDA 图形资源以获取 D3D11 缓冲区指针
    cudaError_t cuda_err = cudaGraphicsMapResources(1, reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to map CUDA graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        d3d_context->Release();
        return false;
    }
    
    void* d3d_buffer_ptr = nullptr;
    size_t buffer_size = 0;
    cuda_err = cudaGraphicsResourceGetMappedPointer(&d3d_buffer_ptr, &buffer_size, 
        static_cast<cudaGraphicsResource*>(cuda_graphics_resource_));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to get mapped CUDA pointer: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(1, reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 0);
        d3d_context->Release();
        return false;
    }
    
    // 直接从 CUDA 输出缓冲区拷贝到映射的 D3D11 缓冲区（设备到设备拷贝）
    size_t total_bytes = width_ * height_ * 4 * sizeof(float);
    cuda_err = cudaMemcpy(d3d_buffer_ptr, stereo_frame.cuda_output_buffer, total_bytes, cudaMemcpyDeviceToDevice);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy from CUDA output to D3D11 buffer: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(1, reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 0);
        d3d_context->Release();
        return false;
    }
    
    // 解除映射 CUDA 资源
    cuda_err = cudaGraphicsUnmapResources(1, reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to unmap CUDA graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        d3d_context->Release();
        return false;
    }
    
    // 更新常量缓冲区
    struct ShaderConstants {
        UINT ImageWidth;
        UINT ImageHeight;
        UINT Padding[2];
    };
    
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    HRESULT hr = d3d_context->Map(constant_buffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    if (FAILED(hr)) {
        std::cerr << "Failed to map constant buffer: HRESULT = 0x" << std::hex << hr << std::endl;
        d3d_context->Release();
        return false;
    }
    
    ShaderConstants* constants = static_cast<ShaderConstants*>(mapped_resource.pData);
    constants->ImageWidth = width_;
    constants->ImageHeight = height_;
    constants->Padding[0] = 0;
    constants->Padding[1] = 0;
    d3d_context->Unmap(constant_buffer_.Get(), 0);
    
    // 设置 compute shader 和资源（纯 D3D11 操作）
    d3d_context->CSSetShader(compute_shader_.Get(), nullptr, 0);
    d3d_context->CSSetConstantBuffers(0, 1, constant_buffer_.GetAddressOf());
    d3d_context->CSSetShaderResources(0, 1, input_srv_.GetAddressOf());
    
    ID3D11UnorderedAccessView* uavs[] = { output_uav_.Get() };
    d3d_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    
    // 分发 compute shader（纯 D3D11，绝不使用 CUDA！）
    UINT group_count_x = (width_ + 15) / 16;
    UINT group_count_y = (height_ + 15) / 16;
    d3d_context->Dispatch(group_count_x, group_count_y, 1);
    
    // 清理绑定
    ID3D11ShaderResourceView* null_srvs[1] = { nullptr };
    ID3D11UnorderedAccessView* null_uavs[1] = { nullptr };
    d3d_context->CSSetShaderResources(0, 1, null_srvs);
    d3d_context->CSSetUnorderedAccessViews(0, 1, null_uavs, nullptr);
    d3d_context->CSSetShader(nullptr, nullptr, 0);
    
    d3d_context->Release();
    
    return true;
}

void D3dStereoVideoDecoder::cleanupResources() {
    // 清理 CUDA 互操作资源
    if (cuda_graphics_resource_) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(cuda_graphics_resource_));
        cuda_graphics_resource_ = nullptr;
    }
    
    // 清理 D3D11 资源
    compute_shader_.Reset();
    input_buffer_.Reset();
    input_srv_.Reset();
    output_uav_.Reset();
    constant_buffer_.Reset();
    d3d_texture_.Reset();
    
    d3d_initialized_ = false;
}