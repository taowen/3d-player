#include "float_rgb_video_decoder.h"
#include <iostream>

FloatRgbVideoDecoder::FloatRgbVideoDecoder()
    : rgb_decoder_(std::make_unique<RgbVideoDecoder>())
    , video_width_(0)
    , video_height_(0) {
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
    
    // 读取RGB解码帧
    if (!rgb_decoder_->readNextFrame(frame.rgb_frame)) {
        return false;
    }
    
    // 转换为浮点格式
    if (!convertToFloat(frame.rgb_frame.rgb_texture, frame.float_texture)) {
        std::cerr << "Failed to convert RGB to float format" << std::endl;
        return false;
    }
    
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

bool FloatRgbVideoDecoder::initializeConversionResources() {
    // 创建浮点输出纹理
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = video_width_;
    desc.Height = video_height_;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;
    
    HRESULT hr = device_->CreateTexture2D(&desc, nullptr, &float_texture_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create float texture, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建UAV用于输出
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uav_desc.Texture2D.MipSlice = 0;
    
    hr = device_->CreateUnorderedAccessView(float_texture_.Get(), &uav_desc, &output_uav_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output UAV, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建简单的计算着色器用于格式转换
    const char* shader_source = R"(
        Texture2D<float4> InputTexture : register(t0);
        RWTexture2D<float4> OutputTexture : register(u0);
        
        [numthreads(8, 8, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            if (id.x >= 1280 || id.y >= 720) return;
            
            // 从BGRA8转换为浮点RGBA
            float4 bgra = InputTexture[id.xy];
            // 交换B和R通道，并确保范围在[0,1]
            float4 rgba = float4(bgra.z, bgra.y, bgra.x, bgra.w);
            OutputTexture[id.xy] = rgba;
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
    
    return true;
}

bool FloatRgbVideoDecoder::convertToFloat(ComPtr<ID3D11Texture2D> rgb_texture, ComPtr<ID3D11Texture2D>& float_texture) {
    if (!rgb_texture || !float_texture_ || !conversion_shader_) {
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
    
    // 设置计算着色器和资源
    context_->CSSetShader(conversion_shader_.Get(), nullptr, 0);
    context_->CSSetShaderResources(0, 1, input_srv.GetAddressOf());
    context_->CSSetUnorderedAccessViews(0, 1, output_uav_.GetAddressOf(), nullptr);
    
    // 分发计算
    UINT dispatch_x = (video_width_ + 7) / 8;
    UINT dispatch_y = (video_height_ + 7) / 8;
    context_->Dispatch(dispatch_x, dispatch_y, 1);
    
    // 清理资源绑定
    ID3D11ShaderResourceView* null_srv = nullptr;
    ID3D11UnorderedAccessView* null_uav = nullptr;
    context_->CSSetShaderResources(0, 1, &null_srv);
    context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    context_->CSSetShader(nullptr, nullptr, 0);
    
    float_texture = float_texture_;
    return true;
}

void FloatRgbVideoDecoder::cleanup() {
    input_srv_.Reset();
    output_uav_.Reset();
    conversion_shader_.Reset();
    float_texture_.Reset();
    context_.Reset();
    device_.Reset();
}