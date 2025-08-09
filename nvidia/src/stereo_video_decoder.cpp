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
    
    
    is_open_ = true;
    std::cout << "Stereo video decoder opened successfully: " << filepath << std::endl;
    return true;
}

void StereoVideoDecoder::close() {
    if (is_open_) {
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
    
    if (!convertToStereo(current_input_frame_.cuda_buffer)) {
        std::cerr << "Failed to convert frame to stereo" << std::endl;
        return false;
    }
    
    frame.input_frame = &current_input_frame_;  // 提供输入帧的引用
    frame.is_valid = true;
    
    // 填充 CUDA 原始输出信息
    frame.cuda_output_buffer = device_output_;
    // 冗余字段已清除，尺寸通过 getWidth()/getHeight() 获取
    
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
    std::string onnx_path = "../stereo_module_half_sbs.onnx";
    
    // 尝试从缓存加载引擎
    tensorrt_engine_ = TensorRTUtils::loadEngineFromCache(static_cast<nvinfer1::IRuntime*>(tensorrt_runtime_), trt_cache_path);
    
    // 如果缓存不存在，从ONNX构建引擎
    if (!tensorrt_engine_) {
        std::cout << "TensorRT cache not found: " << trt_cache_path << std::endl;
        std::cout << "Building TensorRT engine from ONNX: " << onnx_path << std::endl;
        
        tensorrt_engine_ = TensorRTUtils::buildEngineFromONNX(*static_cast<TensorRTUtils::Logger*>(tensorrt_logger_), onnx_path, trt_cache_path);
        if (!tensorrt_engine_) {
            std::cerr << "Failed to build TensorRT engine from ONNX" << std::endl;
            return false;
        }
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

bool StereoVideoDecoder::convertToStereo(void* input_cuda_ptr) {
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
    
    if (!prepareInferenceInput(input_cuda_ptr, runtime_dims)) {
        return false;
    }
    
    if (!executeInference()) {
        return false;
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