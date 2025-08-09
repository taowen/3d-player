#include "float_rgb_video_decoder.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

FloatRgbVideoDecoder::FloatRgbVideoDecoder()
    : rgb_decoder_(std::make_unique<RgbVideoDecoder>())
    , video_width_(0)
    , video_height_(0)
    , cuda_graphics_resource_(nullptr)
    , cuda_device_ptr_(nullptr)
    , buffer_size_(0) {
}

FloatRgbVideoDecoder::~FloatRgbVideoDecoder() {
    close();
}

bool FloatRgbVideoDecoder::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    // 打开RGB解码器
    if (!rgb_decoder_->open(filepath)) {
        std::cerr << "Failed to open RGB decoder for: " << filepath << std::endl;
        return false;
    }
    
    video_width_ = rgb_decoder_->getWidth();
    video_height_ = rgb_decoder_->getHeight();
    
    // 获取D3D11设备
    auto hw_decoder = rgb_decoder_->getHwDecoder();
    if (!hw_decoder) {
        std::cerr << "Failed to get hardware decoder" << std::endl;
        close();
        return false;
    }
    
    device_ = hw_decoder->getD3D11Device();
    if (!device_) {
        std::cerr << "Failed to get D3D11 device" << std::endl;
        close();
        return false;
    }
    
    device_->GetImmediateContext(&context_);
    
    // 初始化格式转换资源
    if (!initializeConversionResources()) {
        std::cerr << "Failed to initialize conversion resources" << std::endl;
        close();
        return false;
    }
    
    return true;
}

bool FloatRgbVideoDecoder::readNextFrame(DecodedFloatRgbFrame& frame) {
    if (!isOpen()) {
        return false;
    }
    
    frame.is_valid = false;
    frame.cuda_buffer = nullptr;
    frame.buffer_size = 0;
    
    // 读取RGB解码帧
    if (!rgb_decoder_->readNextFrame(frame.rgb_frame)) {
        return false;
    }
    
    // 转换为CUDA兼容的浮点缓冲区
    if (!convertToCudaBuffer(frame.rgb_frame.rgb_texture, frame.cuda_buffer)) {
        std::cerr << "Failed to convert RGB to CUDA buffer" << std::endl;
        return false;
    }
    
    frame.buffer_size = buffer_size_;
    frame.is_valid = true;
    return true;
}

bool FloatRgbVideoDecoder::isOpen() const {
    return rgb_decoder_ && rgb_decoder_->isOpen();
}

bool FloatRgbVideoDecoder::isEOF() const {
    return rgb_decoder_ ? rgb_decoder_->isEOF() : true;
}

void FloatRgbVideoDecoder::close() {
    cleanup();
    if (rgb_decoder_) {
        rgb_decoder_->close();
    }
    video_width_ = 0;
    video_height_ = 0;
}

RgbVideoDecoder* FloatRgbVideoDecoder::getRgbDecoder() const {
    return rgb_decoder_.get();
}

int FloatRgbVideoDecoder::getWidth() const {
    return video_width_;
}

int FloatRgbVideoDecoder::getHeight() const {
    return video_height_;
}

void* FloatRgbVideoDecoder::getCudaBuffer() const {
    return cuda_device_ptr_;
}

size_t FloatRgbVideoDecoder::getBufferSize() const {
    return buffer_size_;
}

bool FloatRgbVideoDecoder::initializeConversionResources() {
    // 计算缓冲区大小：BCHW格式, 4通道(RGBA), float32
    // B=1, C=4, H=video_height_, W=video_width_
    buffer_size_ = 1 * 4 * video_height_ * video_width_ * sizeof(float);
    
    // 创建D3D11 Buffer用作CUDA互操作，线性存储BCHW格式数据
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.ByteWidth = static_cast<UINT>(buffer_size_);
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    buffer_desc.CPUAccessFlags = 0;
    buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    buffer_desc.StructureByteStride = 0;
    
    HRESULT hr = device_->CreateBuffer(&buffer_desc, nullptr, &cuda_buffer_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create CUDA buffer, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建UAV用于计算着色器输出
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.FirstElement = 0;
    uav_desc.Buffer.NumElements = static_cast<UINT>(buffer_size_ / sizeof(float));
    uav_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    
    hr = device_->CreateUnorderedAccessView(cuda_buffer_.Get(), &uav_desc, &output_uav_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output UAV, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 注册CUDA图形资源
    cudaError_t cuda_err = cudaGraphicsD3D11RegisterResource(
        reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 
        cuda_buffer_.Get(), 
        cudaGraphicsRegisterFlagsNone);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to register CUDA graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    // 创建计算着色器用于格式转换：BGRA8 -> BCHW float32 RGBA
    const char* shader_source = R"(
        Texture2D<float4> InputTexture : register(t0);
        RWByteAddressBuffer OutputBuffer : register(u0);
        
        cbuffer Constants : register(b0)
        {
            uint width;
            uint height;
        };
        
        [numthreads(8, 8, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            if (id.x >= width || id.y >= height) return;
            
            // 读取BGRA8像素并直接使用，无需通道交换
            // 因为测试表明输入已经是正确的RGB格式
            float4 rgba = InputTexture[id.xy];
            
            // 计算BCHW布局中的偏移量
            // B=1, C=4, H=height, W=width
            // BCHW布局：[R通道整个图像] [G通道整个图像] [B通道整个图像] [A通道整个图像]
            uint pixel_idx = id.y * width + id.x;
            uint channel_size = width * height;
            
            // 按BCHW格式写入：R, G, B, A通道分别连续存储
            uint r_offset = (0 * channel_size + pixel_idx) * 4;  // R通道偏移
            uint g_offset = (1 * channel_size + pixel_idx) * 4;  // G通道偏移  
            uint b_offset = (2 * channel_size + pixel_idx) * 4;  // B通道偏移
            uint a_offset = (3 * channel_size + pixel_idx) * 4;  // A通道偏移
            
            OutputBuffer.Store(r_offset, asuint(rgba.x));  // R
            OutputBuffer.Store(g_offset, asuint(rgba.y));  // G  
            OutputBuffer.Store(b_offset, asuint(rgba.z));  // B
            OutputBuffer.Store(a_offset, asuint(rgba.w));  // A
        }
    )";
    
    ComPtr<ID3DBlob> shader_blob;
    ComPtr<ID3DBlob> error_blob;
    hr = D3DCompile(shader_source, strlen(shader_source), nullptr, nullptr, nullptr,
                    "main", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, 
                    &shader_blob, &error_blob);
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Shader compilation error: " << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        std::cerr << "Failed to compile conversion shader, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    hr = device_->CreateComputeShader(shader_blob->GetBufferPointer(), 
                                     shader_blob->GetBufferSize(), 
                                     nullptr, &conversion_shader_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建常量缓冲区用于传递width和height
    struct Constants {
        UINT width;
        UINT height;
        UINT padding[2];  // 保持16字节对齐
    };
    
    D3D11_BUFFER_DESC cb_desc = {};
    cb_desc.ByteWidth = sizeof(Constants);
    cb_desc.Usage = D3D11_USAGE_DYNAMIC;
    cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    cb_desc.MiscFlags = 0;
    cb_desc.StructureByteStride = 0;
    
    Constants init_data = { static_cast<UINT>(video_width_), static_cast<UINT>(video_height_), 0, 0 };
    D3D11_SUBRESOURCE_DATA init_sub_data = {};
    init_sub_data.pSysMem = &init_data;
    
    ComPtr<ID3D11Buffer> constant_buffer;
    hr = device_->CreateBuffer(&cb_desc, &init_sub_data, &constant_buffer);
    if (FAILED(hr)) {
        std::cerr << "Failed to create constant buffer, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 存储常量缓冲区（需要添加到类成员中）
    constant_buffer_ = constant_buffer;
    
    return true;
}

bool FloatRgbVideoDecoder::convertToCudaBuffer(ComPtr<ID3D11Texture2D> rgb_texture, void*& cuda_ptr) {
    if (!rgb_texture || !cuda_buffer_ || !conversion_shader_ || !constant_buffer_) {
        return false;
    }
    
    // 创建输入纹理的SRV
    ComPtr<ID3D11ShaderResourceView> input_srv;
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = 1;
    
    HRESULT hr = device_->CreateShaderResourceView(rgb_texture.Get(), &srv_desc, &input_srv);
    if (FAILED(hr)) {
        std::cerr << "Failed to create input SRV, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 映射CUDA资源以获取设备指针
    cudaError_t cuda_err = cudaGraphicsMapResources(1, reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to map CUDA graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    size_t mapped_size;
    cuda_err = cudaGraphicsResourceGetMappedPointer(&cuda_device_ptr_, &mapped_size, 
        static_cast<cudaGraphicsResource*>(cuda_graphics_resource_));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to get mapped CUDA pointer: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(1, reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 0);
        return false;
    }
    
    // 设置计算着色器和资源
    context_->CSSetShader(conversion_shader_.Get(), nullptr, 0);
    context_->CSSetShaderResources(0, 1, input_srv.GetAddressOf());
    context_->CSSetUnorderedAccessViews(0, 1, output_uav_.GetAddressOf(), nullptr);
    context_->CSSetConstantBuffers(0, 1, constant_buffer_.GetAddressOf());
    
    // 分发计算
    UINT dispatch_x = (video_width_ + 7) / 8;
    UINT dispatch_y = (video_height_ + 7) / 8;
    context_->Dispatch(dispatch_x, dispatch_y, 1);
    
    // 清理资源绑定
    ID3D11ShaderResourceView* null_srv = nullptr;
    ID3D11UnorderedAccessView* null_uav = nullptr;
    ID3D11Buffer* null_cb = nullptr;
    context_->CSSetShaderResources(0, 1, &null_srv);
    context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    context_->CSSetConstantBuffers(0, 1, &null_cb);
    context_->CSSetShader(nullptr, nullptr, 0);
    
    // 取消映射CUDA资源
    cuda_err = cudaGraphicsUnmapResources(1, reinterpret_cast<cudaGraphicsResource**>(&cuda_graphics_resource_), 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to unmap CUDA graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    cuda_ptr = cuda_device_ptr_;
    return true;
}

void FloatRgbVideoDecoder::cleanup() {
    // 清理CUDA资源
    if (cuda_graphics_resource_) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(cuda_graphics_resource_));
        cuda_graphics_resource_ = nullptr;
    }
    cuda_device_ptr_ = nullptr;
    buffer_size_ = 0;
    
    // 清理D3D11资源
    input_srv_.Reset();
    output_uav_.Reset();
    conversion_shader_.Reset();
    cuda_buffer_.Reset();
    constant_buffer_.Reset();
    context_.Reset();
    device_.Reset();
}