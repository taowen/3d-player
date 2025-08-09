#include "tensorrt_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <wrl/client.h>
#include <dxgi.h>
#include <NvOnnxParser.h>

namespace TensorRTUtils {

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

bool initializeCudaDevice(void* d3d11_device, int& cuda_device) {
    ID3D11Device* device = static_cast<ID3D11Device*>(d3d11_device);
    if (!device) {
        std::cerr << "Failed to get D3D11 device" << std::endl;
        return false;
    }
    
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;
    HRESULT hr = device->QueryInterface(__uuidof(IDXGIDevice), &dxgi_device);
    if (FAILED(hr)) {
        std::cerr << "Failed to get DXGI device from D3D11 device" << std::endl;
        return false;
    }
    
    Microsoft::WRL::ComPtr<IDXGIAdapter> dxgi_adapter;
    hr = dxgi_device->GetAdapter(&dxgi_adapter);
    if (FAILED(hr)) {
        std::cerr << "Failed to get DXGI adapter" << std::endl;
        return false;
    }
    
    cuda_device = -1;
    cudaError_t cuda_err = cudaD3D11GetDevice(&cuda_device, dxgi_adapter.Get());
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device from D3D11 adapter: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    cuda_err = cudaSetDevice(cuda_device);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << cuda_device << ": " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    std::cout << "Successfully set CUDA device to " << cuda_device << std::endl;
    return true;
}

nvinfer1::IRuntime* createTensorRTRuntime(nvinfer1::ILogger& logger) {
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return nullptr;
    }
    return runtime;
}

nvinfer1::ICudaEngine* loadEngineFromCache(nvinfer1::IRuntime* runtime, const std::string& cache_path) {
    std::ifstream cache_file(cache_path, std::ios::binary);
    if (!cache_file.is_open()) {
        std::cerr << "TensorRT cache not found: " << cache_path << std::endl;
        return nullptr;
    }
    
    cache_file.seekg(0, std::ios::end);
    size_t cache_size = cache_file.tellg();
    cache_file.seekg(0, std::ios::beg);
    
    std::vector<char> cache_data(cache_size);
    cache_file.read(cache_data.data(), cache_size);
    cache_file.close();
    
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(cache_data.data(), cache_size);
    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
        return nullptr;
    }
    
    return engine;
}

nvinfer1::ICudaEngine* buildEngineFromONNX(nvinfer1::ILogger& logger, const std::string& onnx_path, const std::string& cache_path) {
    // 检查ONNX文件是否存在
    std::ifstream onnx_file(onnx_path, std::ios::binary);
    if (!onnx_file.good()) {
        std::cerr << "ONNX file not found: " << onnx_path << std::endl;
        return nullptr;
    }
    onnx_file.close();
    
    std::cout << "Building TensorRT engine from ONNX: " << onnx_path << std::endl;
    
    // 使用智能指针管理TensorRT对象
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return nullptr;
    }
    
    // 创建网络定义
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    if (!network) {
        std::cerr << "Failed to create network definition" << std::endl;
        return nullptr;
    }
    
    // 创建ONNX解析器
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return nullptr;
    }
    
    // 解析ONNX文件
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file: " << onnx_path << std::endl;
        return nullptr;
    }
    
    // 创建构建配置
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return nullptr;
    }
    
    // 启用FP16优化
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Using FP16 precision" << std::endl;
    }
    
    // 设置内存池大小 (2GB for 4K support)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2U << 30);
    
    // 设置动态形状配置文件
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Failed to create optimization profile" << std::endl;
        return nullptr;
    }
    
    // 获取输入张量信息
    nvinfer1::ITensor* input = network->getInput(0);
    nvinfer1::Dims input_dims = input->getDimensions();
    
    // 设置动态形状范围 - 支持从 480p 到 1440p 分辨率 (平衡构建时间和实用性)
    nvinfer1::Dims min_dims = input_dims;
    nvinfer1::Dims opt_dims = input_dims;  
    nvinfer1::Dims max_dims = input_dims;
    
    // 批次大小固定为1
    min_dims.d[0] = 1;
    opt_dims.d[0] = 1;
    max_dims.d[0] = 1;
    
    // 通道数固定为4 (RGBA)
    min_dims.d[1] = 4;
    opt_dims.d[1] = 4;
    max_dims.d[1] = 4;
    
    // 高度范围: 480 - 1440 (平衡构建时间)
    min_dims.d[2] = 480;
    opt_dims.d[2] = 720;   // 优化目标720p
    max_dims.d[2] = 1440;  // 最大1440p
    
    // 宽度范围: 854 - 2560 (平衡构建时间)  
    min_dims.d[3] = 854;
    opt_dims.d[3] = 1280;  // 优化目标1280
    max_dims.d[3] = 2560;  // 最大2560
    
    if (!profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims)) {
        std::cerr << "Failed to set optimization profile dimensions" << std::endl;
        return nullptr;
    }
    
    config->addOptimizationProfile(profile);
    
    std::cout << "Building TensorRT engine with 4K support..." << std::endl;
    
    // 构建引擎
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return nullptr;
    }
    
    // 保存引擎缓存
    std::ofstream cache_file(cache_path, std::ios::binary);
    if (cache_file.is_open()) {
        cache_file.write(static_cast<const char*>(plan->data()), plan->size());
        cache_file.close();
        std::cout << "TensorRT engine cached to: " << cache_path << std::endl;
    } else {
        std::cerr << "Warning: Failed to save TensorRT cache to: " << cache_path << std::endl;
    }
    
    // 反序列化引擎
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(createTensorRTRuntime(logger));
    if (!runtime) {
        return nullptr;
    }
    
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    
    if (!engine) {
        std::cerr << "Failed to deserialize built engine" << std::endl;
        return nullptr;
    }
    
    std::cout << "TensorRT engine built successfully" << std::endl;
    return engine;
}

}