#include "d3d_stereo_video_decoder.h"
#include <iostream>

D3dStereoVideoDecoder::D3dStereoVideoDecoder() 
    : stereo_decoder_(std::make_unique<StereoVideoDecoder>())
    , interop_initialized_(false)
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
    
    return initializeCudaD3DInterop();
}

void D3dStereoVideoDecoder::close() {
    cleanupInterop();
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

bool D3dStereoVideoDecoder::initializeCudaD3DInterop() {
    if (interop_initialized_) {
        return true;
    }
    
    if (width_ <= 0 || height_ <= 0) {
        std::cerr << "Invalid video dimensions for D3D interop: " << width_ << "x" << height_ << std::endl;
        return false;
    }
    
    ID3D11Device* d3d_device = getD3D11Device();
    if (!d3d_device) {
        std::cerr << "No D3D11 device available for interop" << std::endl;
        return false;
    }
    
    // 创建 D3D11 纹理用于立体输出
    // 格式：RGBA32F，尺寸与视频相同
    D3D11_TEXTURE2D_DESC texture_desc = {};
    texture_desc.Width = width_;
    texture_desc.Height = height_;
    texture_desc.MipLevels = 1;
    texture_desc.ArraySize = 1;
    texture_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.Usage = D3D11_USAGE_DEFAULT;
    texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texture_desc.CPUAccessFlags = 0;
    texture_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;  // CUDA interop 需要共享标志
    
    HRESULT hr = d3d_device->CreateTexture2D(&texture_desc, nullptr, &d3d_texture_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 texture for CUDA interop: HRESULT = 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 注册 D3D11 纹理到 CUDA
    cudaError_t cuda_err = cudaGraphicsD3D11RegisterResource(
        &cuda_graphics_resource_, 
        d3d_texture_.Get(), 
        cudaGraphicsRegisterFlagsNone
    );
    
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to register D3D11 texture to CUDA: " << cudaGetErrorString(cuda_err) << std::endl;
        d3d_texture_.Reset();
        return false;
    }
    
    interop_initialized_ = true;
    std::cout << "CUDA-D3D11 interop initialized successfully: " << width_ << "x" << height_ << " RGBA32F" << std::endl;
    
    return true;
}

bool D3dStereoVideoDecoder::convertCudaToD3DTexture(const DecodedStereoFrame& stereo_frame) {
    if (!interop_initialized_ || !cuda_graphics_resource_) {
        std::cerr << "CUDA-D3D11 interop not initialized" << std::endl;
        return false;
    }
    
    if (!stereo_frame.is_valid || !stereo_frame.cuda_output_buffer) {
        std::cerr << "Invalid stereo frame for D3D conversion" << std::endl;
        return false;
    }
    
    // 映射 CUDA 资源
    cudaError_t cuda_err = cudaGraphicsMapResources(1, &cuda_graphics_resource_, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to map CUDA graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    // 获取映射后的 CUDA 数组
    cudaArray_t cuda_array = nullptr;
    cuda_err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_graphics_resource_, 0, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to get mapped CUDA array: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_graphics_resource_, 0);
        return false;
    }
    
    // 从 CUDA 线性内存拷贝到纹理数组
    // StereoVideoDecoder 输出格式：BCHW 平面存储的 float32 RGBA
    size_t pixel_size = 4 * sizeof(float);  // RGBA32F
    size_t pitch = width_ * pixel_size;
    
    cuda_err = cudaMemcpy2DToArray(
        cuda_array,
        0, 0,  // offset
        stereo_frame.cuda_output_buffer,
        pitch,  // src pitch
        pitch,  // width in bytes
        height_,  // height
        cudaMemcpyDeviceToDevice
    );
    
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy CUDA data to texture array: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_graphics_resource_, 0);
        return false;
    }
    
    // 解除映射
    cuda_err = cudaGraphicsUnmapResources(1, &cuda_graphics_resource_, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to unmap CUDA graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    return true;
}

void D3dStereoVideoDecoder::cleanupInterop() {
    if (cuda_graphics_resource_) {
        cudaGraphicsUnregisterResource(cuda_graphics_resource_);
        cuda_graphics_resource_ = nullptr;
    }
    
    d3d_texture_.Reset();
    interop_initialized_ = false;
}