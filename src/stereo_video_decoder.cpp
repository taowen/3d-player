#include "stereo_video_decoder.h"
#include "float_rgb_video_decoder.h"
#include "tensorrt_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

extern "C" {
#include <libavutil/rational.h>
}

namespace {
    static constexpr size_t RGBA_FLOAT_BYTES = 16; // R32G32B32A32_FLOAT
}


StereoVideoDecoder::StereoVideoDecoder() 
    : float_rgb_decoder_(std::make_unique<FloatRgbVideoDecoder>())
    , is_open_(false)
    , device_input_size_(0)
    , device_output_size_(0)
    , registered_input_texture_(nullptr)
    , registered_output_texture_(nullptr)
    , tensorrt_initialized_(false) {
}

StereoVideoDecoder::~StereoVideoDecoder() {
    close();
}

bool StereoVideoDecoder::open(const std::string& filepath) {
    close();
    
    if (!float_rgb_decoder_->open(filepath)) {
        std::cerr << "Failed to open video file: " << filepath << std::endl;
        return false;
    }
    
    if (!initializeTensorRT()) {
        std::cerr << "Failed to initialize TensorRT processor" << std::endl;
        float_rgb_decoder_->close();
        return false;
    }
    
    ID3D11Device* device = float_rgb_decoder_->getRgbDecoder()->getHwDecoder()->getD3D11Device();
    D3D11_TEXTURE2D_DESC stereo_desc = {};
    stereo_desc.Width = float_rgb_decoder_->getWidth();
    stereo_desc.Height = float_rgb_decoder_->getHeight();
    stereo_desc.MipLevels = 1;
    stereo_desc.ArraySize = 1;
    stereo_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    stereo_desc.SampleDesc.Count = 1;
    stereo_desc.Usage = D3D11_USAGE_DEFAULT;
    stereo_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    stereo_desc.CPUAccessFlags = 0;
    stereo_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    HRESULT hr = device->CreateTexture2D(&stereo_desc, nullptr, &stereo_texture_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create stereo texture" << std::endl;
        cleanupTensorRT();
        float_rgb_decoder_->close();
        return false;
    }
    
    is_open_ = true;
    std::cout << "Stereo video decoder opened successfully: " << filepath << std::endl;
    return true;
}

void StereoVideoDecoder::close() {
    if (is_open_) {
        stereo_texture_.Reset();
        cleanupTensorRT();
        float_rgb_decoder_->close();
        is_open_ = false;
    }
}

bool StereoVideoDecoder::isOpen() const {
    return is_open_ && float_rgb_decoder_->isOpen();
}

bool StereoVideoDecoder::isEOF() const {
    return float_rgb_decoder_->isEOF();
}

bool StereoVideoDecoder::readNextFrame(DecodedStereoFrame& frame) {
    if (!isOpen()) {
        return false;
    }
    
    FloatRgbVideoDecoder::DecodedFloatRgbFrame float_rgb_frame;
    if (!float_rgb_decoder_->readNextFrame(float_rgb_frame)) {
        return false;
    }
    
    if (!convertToStereo(float_rgb_frame.float_texture.Get(), stereo_texture_.Get())) {
        std::cerr << "Failed to convert frame to stereo" << std::endl;
        return false;
    }
    
    frame.stereo_texture = stereo_texture_;
    frame.pts_seconds = float_rgb_frame.rgb_frame.hw_frame.frame->pts * av_q2d(float_rgb_decoder_->getRgbDecoder()->getHwDecoder()->getStreamReader()->getStreamInfo().video_time_base);
    frame.is_valid = true;
    
    return true;
}

int StereoVideoDecoder::getWidth() const {
    return float_rgb_decoder_->getWidth();
}

int StereoVideoDecoder::getHeight() const {
    return float_rgb_decoder_->getHeight();
}

ID3D11Device* StereoVideoDecoder::getD3D11Device() const {
    return float_rgb_decoder_->getRgbDecoder()->getHwDecoder()->getD3D11Device();
}

bool StereoVideoDecoder::initializeTensorRT() {
    if (tensorrt_initialized_) {
        return true;
    }
    
    ID3D11Device* device = float_rgb_decoder_->getRgbDecoder()->getHwDecoder()->getD3D11Device();
    if (!device) {
        std::cerr << "Failed to get D3D11 device" << std::endl;
        return false;
    }
    
    int cuda_device;
    if (!TensorRTUtils::initializeCudaDevice(device, cuda_device)) {
        return false;
    }
    
    // 创建 Logger
    tensorrt_logger_ = new TensorRTUtils::Logger();
    
    // 创建 TensorRT runtime
    tensorrt_runtime_ = TensorRTUtils::createTensorRTRuntime(*static_cast<TensorRTUtils::Logger*>(tensorrt_logger_));
    if (!tensorrt_runtime_) {
        return false;
    }
    
    std::string trt_cache_path = "../stereo_module_half_sbs_fp16.trt";
    tensorrt_engine_ = TensorRTUtils::loadEngineFromCache(static_cast<nvinfer1::IRuntime*>(tensorrt_runtime_), trt_cache_path);
    if (!tensorrt_engine_) {
        return false;
    }
    
    tensorrt_context_ = static_cast<nvinfer1::ICudaEngine*>(tensorrt_engine_)->createExecutionContext();
    if (!tensorrt_context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    tensorrt_initialized_ = true;
    std::cout << "TensorRT processor initialized successfully" << std::endl;
    return true;
}

bool StereoVideoDecoder::convertToStereo(ID3D11Texture2D* input_rgb, ID3D11Texture2D* output_stereo) {
    if (!tensorrt_initialized_) {
        return false;
    }
    
    D3D11_TEXTURE2D_DESC input_desc;
    input_rgb->GetDesc(&input_desc);
    
    TensorDims runtime_dims;
    runtime_dims.nbDims = 4;
    runtime_dims.d[0] = 1;
    runtime_dims.d[1] = 4;
    runtime_dims.d[2] = (int)input_desc.Height;
    runtime_dims.d[3] = (int)input_desc.Width;
    
    if (!registerCudaResources(input_rgb, output_stereo)) {
        return false;
    }
    
    if (!prepareInferenceInput(input_rgb, runtime_dims)) {
        return false;
    }
    
    if (!executeInference()) {
        return false;
    }
    
    if (!copyInferenceOutput(output_stereo, input_desc)) {
        return false;
    }
    
    return true;
}


bool StereoVideoDecoder::registerCudaResources(ID3D11Texture2D* input_rgb, ID3D11Texture2D* output_stereo) {
    cudaError_t cuda_err;
    
    // 检查输入纹理是否需要重新注册
    if (input_resource_ == nullptr || registered_input_texture_ != input_rgb) {
        if (input_resource_ != nullptr) {
            cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(input_resource_));
            input_resource_ = nullptr;
            registered_input_texture_ = nullptr;
        }
        
        cuda_err = cudaGraphicsD3D11RegisterResource(
            reinterpret_cast<cudaGraphicsResource**>(&input_resource_), input_rgb, cudaGraphicsRegisterFlagsNone);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Failed to register input texture: " << cudaGetErrorString(cuda_err) << std::endl;
            return false;
        }
        registered_input_texture_ = input_rgb;
    }
    
    // 检查输出纹理是否需要重新注册
    if (output_resource_ == nullptr || registered_output_texture_ != output_stereo) {
        if (output_resource_ != nullptr) {
            cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(output_resource_));
            output_resource_ = nullptr;
            registered_output_texture_ = nullptr;
        }
        
        cuda_err = cudaGraphicsD3D11RegisterResource(
            reinterpret_cast<cudaGraphicsResource**>(&output_resource_), output_stereo, cudaGraphicsRegisterFlagsNone);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Failed to register output texture: " << cudaGetErrorString(cuda_err) << std::endl;
            // 清理输入资源防止不一致状态
            if (input_resource_ != nullptr) {
                cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(input_resource_));
                input_resource_ = nullptr;
                registered_input_texture_ = nullptr;
            }
            return false;
        }
        registered_output_texture_ = output_stereo;
    }
    
    return true;
}

bool StereoVideoDecoder::prepareInferenceInput(ID3D11Texture2D* input_rgb, const TensorDims& runtime_dims) {
    const char* input_tensor_name = static_cast<nvinfer1::ICudaEngine*>(tensorrt_engine_)->getIOTensorName(0);
    
    // 转换为 TensorRT Dims
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = runtime_dims.nbDims;
    for (int i = 0; i < runtime_dims.nbDims; ++i) {
        trt_dims.d[i] = runtime_dims.d[i];
    }
    
    if (!static_cast<nvinfer1::IExecutionContext*>(tensorrt_context_)->setInputShape(input_tensor_name, trt_dims)) {
        std::cerr << "Failed to set runtime input shape" << std::endl;
        return false;
    }
    
    D3D11_TEXTURE2D_DESC input_desc;
    input_rgb->GetDesc(&input_desc);
    
    size_t bytes_per_pixel = RGBA_FLOAT_BYTES;
    
    cudaGraphicsResource* resources[] = {static_cast<cudaGraphicsResource*>(input_resource_), static_cast<cudaGraphicsResource*>(output_resource_)};
    cudaError_t cuda_err = cudaGraphicsMapResources(2, resources, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to map graphics resources: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    cudaArray_t input_array;
    cuda_err = cudaGraphicsSubResourceGetMappedArray(&input_array, static_cast<cudaGraphicsResource*>(input_resource_), 0, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to get mapped array from input resource: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(2, resources, 0);
        return false;
    }
    
    size_t input_size = input_desc.Width * input_desc.Height * RGBA_FLOAT_BYTES;
    size_t output_size = input_desc.Width * input_desc.Height * RGBA_FLOAT_BYTES;
    
    // 检查输入内存是否需要重新分配
    if (device_input_ == nullptr || device_input_size_ != input_size) {
        if (device_input_ != nullptr) {
            cudaFree(device_input_);
            device_input_ = nullptr;
        }
        cuda_err = cudaMalloc(&device_input_, input_size);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Failed to allocate input CUDA memory: " << cudaGetErrorString(cuda_err) << std::endl;
            cudaGraphicsUnmapResources(2, resources, 0);
            return false;
        }
        device_input_size_ = input_size;
    }
    
    // 检查输出内存是否需要重新分配
    if (device_output_ == nullptr || device_output_size_ != output_size) {
        if (device_output_ != nullptr) {
            cudaFree(device_output_);
            device_output_ = nullptr;
        }
        cuda_err = cudaMalloc(&device_output_, output_size);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Failed to allocate output CUDA memory: " << cudaGetErrorString(cuda_err) << std::endl;
            cudaGraphicsUnmapResources(2, resources, 0);
            return false;
        }
        device_output_size_ = output_size;
    }
    
    size_t input_pitch = input_desc.Width * bytes_per_pixel;
    cuda_err = cudaMemcpy2DFromArray(
        device_input_, 
        input_pitch,
        input_array, 0, 0,
        input_pitch,
        input_desc.Height,
        cudaMemcpyDeviceToDevice
    );
    
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy from input array: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(2, resources, 0);
        return false;
    }
    
    return true;
}

bool StereoVideoDecoder::executeInference() {
    void* bindings[] = {device_input_, device_output_};
    bool inference_success = static_cast<nvinfer1::IExecutionContext*>(tensorrt_context_)->executeV2(bindings);
    
    if (!inference_success) {
        std::cerr << "TensorRT inference failed" << std::endl;
        return false;
    }
    
    return true;
}

bool StereoVideoDecoder::copyInferenceOutput(ID3D11Texture2D* output_stereo, const D3D11_TEXTURE2D_DESC& input_desc) {
    cudaGraphicsResource* resources[] = {static_cast<cudaGraphicsResource*>(input_resource_), static_cast<cudaGraphicsResource*>(output_resource_)};
    
    cudaArray_t output_array;
    cudaError_t cuda_err = cudaGraphicsSubResourceGetMappedArray(&output_array, static_cast<cudaGraphicsResource*>(output_resource_), 0, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to get mapped array from output resource: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(2, resources, 0);
        return false;
    }
    
    size_t output_pitch = input_desc.Width * RGBA_FLOAT_BYTES;
    cuda_err = cudaMemcpy2DToArray(
        output_array, 0, 0,
        device_output_,
        output_pitch,
        output_pitch,
        input_desc.Height,
        cudaMemcpyDeviceToDevice
    );
    
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy to output array: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(2, resources, 0);
        return false;
    }
    
    cuda_err = cudaGraphicsUnmapResources(2, resources, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to unmap graphics resources: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    return true;
}

void StereoVideoDecoder::cleanupTensorRT() {
    // 清理 CUDA 内存
    if (device_input_) {
        cudaFree(device_input_);
        device_input_ = nullptr;
        device_input_size_ = 0;
    }
    if (device_output_) {
        cudaFree(device_output_);
        device_output_ = nullptr;
        device_output_size_ = 0;
    }
    
    // 清理 Graphics 资源
    if (input_resource_) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(input_resource_));
        input_resource_ = nullptr;
        registered_input_texture_ = nullptr;
    }
    if (output_resource_) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(output_resource_));
        output_resource_ = nullptr;
        registered_output_texture_ = nullptr;
    }
    
    // 清理 TensorRT 对象（按依赖顺序释放）
    if (tensorrt_context_) {
        delete static_cast<nvinfer1::IExecutionContext*>(tensorrt_context_);
        tensorrt_context_ = nullptr;
    }
    if (tensorrt_engine_) {
        delete static_cast<nvinfer1::ICudaEngine*>(tensorrt_engine_);
        tensorrt_engine_ = nullptr;
    }
    if (tensorrt_runtime_) {
        delete static_cast<nvinfer1::IRuntime*>(tensorrt_runtime_);
        tensorrt_runtime_ = nullptr;
    }
    if (tensorrt_logger_) {
        delete static_cast<TensorRTUtils::Logger*>(tensorrt_logger_);
        tensorrt_logger_ = nullptr;
    }
    
    tensorrt_initialized_ = false;
}