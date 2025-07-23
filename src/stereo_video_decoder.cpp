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
    
    // BMP文件头结构
    #pragma pack(push, 1)
    struct BMPFileHeader {
        uint16_t bfType = 0x4D42;     // "BM"
        uint32_t bfSize;
        uint16_t bfReserved1 = 0;
        uint16_t bfReserved2 = 0;
        uint32_t bfOffBits = 54;      // 文件头+信息头大小
    };

    struct BMPInfoHeader {
        uint32_t biSize = 40;
        int32_t biWidth;
        int32_t biHeight;
        uint16_t biPlanes = 1;
        uint16_t biBitCount = 24;     // 24位RGB
        uint32_t biCompression = 0;   // 无压缩
        uint32_t biSizeImage = 0;
        int32_t biXPelsPerMeter = 0;
        int32_t biYPelsPerMeter = 0;
        uint32_t biClrUsed = 0;
        uint32_t biClrImportant = 0;
    };
    #pragma pack(pop)
    
    // 保存CUDA输出为BMP文件
    void saveCudaOutputToBMP(const std::vector<float>& pixels, uint32_t width, uint32_t height, const std::string& filename) {
        uint32_t row_size = ((width * 3 + 3) / 4) * 4;
        uint32_t image_size = row_size * height;
        
        BMPFileHeader file_header;
        file_header.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + image_size;
        
        BMPInfoHeader info_header;
        info_header.biWidth = width;
        info_header.biHeight = height;
        info_header.biSizeImage = image_size;
        
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
            file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));
            
            std::vector<uint8_t> row_data(row_size, 0);
            for (int y = height - 1; y >= 0; --y) {
                for (uint32_t x = 0; x < width; ++x) {
                    size_t pixel_idx = (y * width + x) * 4;
                    
                    // 浮点值[0,1]转换为8位整数[0,255]，RGBA -> BGR
                    uint8_t r = static_cast<uint8_t>(std::round(pixels[pixel_idx + 0] * 255.0f));
                    uint8_t g = static_cast<uint8_t>(std::round(pixels[pixel_idx + 1] * 255.0f));
                    uint8_t b = static_cast<uint8_t>(std::round(pixels[pixel_idx + 2] * 255.0f));
                    
                    row_data[x * 3 + 0] = b; // B
                    row_data[x * 3 + 1] = g; // G
                    row_data[x * 3 + 2] = r; // R
                }
                file.write(reinterpret_cast<const char*>(row_data.data()), row_size);
            }
            file.close();
        }
    }
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
    
    if (!float_rgb_decoder_->readNextFrame(current_input_frame_)) {
        return false;
    }
    
    if (!convertToStereo(current_input_frame_.cuda_buffer, stereo_texture_.Get())) {
        std::cerr << "Failed to convert frame to stereo" << std::endl;
        return false;
    }
    
    frame.stereo_texture = stereo_texture_;
    frame.frame = current_input_frame_.rgb_frame.hw_frame.frame;
    frame.input_frame = &current_input_frame_;  // 提供输入帧的引用
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

AVRational StereoVideoDecoder::getVideoTimeBase() const {
    if (!float_rgb_decoder_) {
        return {0, 1};
    }
    return float_rgb_decoder_->getRgbDecoder()->getHwDecoder()->getStreamReader()->getStreamInfo().video_time_base;
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

bool StereoVideoDecoder::convertToStereo(void* input_cuda_ptr, ID3D11Texture2D* output_stereo) {
    if (!tensorrt_initialized_ || !input_cuda_ptr) {
        return false;
    }
    
    // 获取视频尺寸用于TensorRT维度设置
    int width = float_rgb_decoder_->getWidth();
    int height = float_rgb_decoder_->getHeight();
    
    TensorDims runtime_dims;
    runtime_dims.nbDims = 4;
    runtime_dims.d[0] = 1;
    runtime_dims.d[1] = 4;
    runtime_dims.d[2] = height;
    runtime_dims.d[3] = width;
    
    if (!registerCudaResources(nullptr, output_stereo)) {
        return false;
    }
    
    if (!prepareInferenceInput(input_cuda_ptr, runtime_dims)) {
        return false;
    }
    
    if (!executeInference()) {
        return false;
    }
    
    D3D11_TEXTURE2D_DESC output_desc;
    output_stereo->GetDesc(&output_desc);
    if (!copyInferenceOutput(output_stereo, output_desc)) {
        return false;
    }
    
    return true;
}


bool StereoVideoDecoder::registerCudaResources(ID3D11Texture2D* input_rgb, ID3D11Texture2D* output_stereo) {
    // 输入已经是CUDA指针，不需要注册输入纹理
    
    // 检查输出纹理是否需要重新注册
    if (output_resource_ == nullptr || registered_output_texture_ != output_stereo) {
        if (output_resource_ != nullptr) {
            cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(output_resource_));
            output_resource_ = nullptr;
            registered_output_texture_ = nullptr;
        }
        
        cudaError_t cuda_err = cudaGraphicsD3D11RegisterResource(
            reinterpret_cast<cudaGraphicsResource**>(&output_resource_), output_stereo, cudaGraphicsRegisterFlagsNone);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Failed to register output texture: " << cudaGetErrorString(cuda_err) << std::endl;
            return false;
        }
        registered_output_texture_ = output_stereo;
    }
    
    return true;
}

bool StereoVideoDecoder::prepareInferenceInput(void* input_cuda_ptr, const TensorDims& runtime_dims) {
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
    
    // 计算输出缓冲区大小
    int width = float_rgb_decoder_->getWidth();
    int height = float_rgb_decoder_->getHeight();
    size_t output_size = width * height * RGBA_FLOAT_BYTES;
    
    // 检查输出内存是否需要重新分配
    if (device_output_ == nullptr || device_output_size_ != output_size) {
        if (device_output_ != nullptr) {
            cudaFree(device_output_);
            device_output_ = nullptr;
        }
        cudaError_t cuda_err = cudaMalloc(&device_output_, output_size);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Failed to allocate output CUDA memory: " << cudaGetErrorString(cuda_err) << std::endl;
            return false;
        }
        device_output_size_ = output_size;
    }
    
    // 直接使用 FloatRgbVideoDecoder 提供的 CUDA 指针作为输入
    device_input_ = input_cuda_ptr;
    
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

bool StereoVideoDecoder::copyInferenceOutput(ID3D11Texture2D* output_stereo, const D3D11_TEXTURE2D_DESC& output_desc) {
    cudaGraphicsResource* output_resource = static_cast<cudaGraphicsResource*>(output_resource_);
    
    // 映射输出纹理资源
    cudaError_t cuda_err = cudaGraphicsMapResources(1, &output_resource, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to map output graphics resource: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    cudaArray_t output_array;
    cuda_err = cudaGraphicsSubResourceGetMappedArray(&output_array, output_resource, 0, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to get mapped array from output resource: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(1, &output_resource, 0);
        return false;
    }
    
    size_t output_pitch = output_desc.Width * RGBA_FLOAT_BYTES;
    
    // 调试：保存CUDA推理输出到BMP (仅前3帧)
    static int debug_frame_count = 0;
    if (debug_frame_count < 3) {
        size_t total_size = output_desc.Width * output_desc.Height * 4 * sizeof(float);
        std::vector<float> host_output(output_desc.Width * output_desc.Height * 4);
        
        cudaError_t copy_err = cudaMemcpy(host_output.data(), device_output_, total_size, cudaMemcpyDeviceToHost);
        if (copy_err == cudaSuccess) {
            saveCudaOutputToBMP(host_output, output_desc.Width, output_desc.Height, 
                              "cuda_output_frame_" + std::to_string(debug_frame_count) + ".bmp");
            std::cout << "Saved CUDA output frame: cuda_output_frame_" << debug_frame_count << ".bmp" << std::endl;
        }
        debug_frame_count++;
    }
    
    cuda_err = cudaMemcpy2DToArray(
        output_array, 0, 0,
        device_output_,
        output_pitch,
        output_pitch,
        output_desc.Height,
        cudaMemcpyDeviceToDevice
    );
    
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy to output array: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaGraphicsUnmapResources(1, &output_resource, 0);
        return false;
    }
    
    cuda_err = cudaGraphicsUnmapResources(1, &output_resource, 0);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to unmap graphics resources: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    return true;
}

void StereoVideoDecoder::cleanupTensorRT() {
    // 清理 CUDA 内存
    // device_input_ 现在指向 FloatRgbVideoDecoder 的内存，不需要我们释放
    device_input_ = nullptr;
    device_input_size_ = 0;
    
    if (device_output_) {
        cudaFree(device_output_);
        device_output_ = nullptr;
        device_output_size_ = 0;
    }
    
    // 清理 Graphics 资源
    // input_resource_ 不再使用，因为输入已经是CUDA指针
    input_resource_ = nullptr;
    registered_input_texture_ = nullptr;
    
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