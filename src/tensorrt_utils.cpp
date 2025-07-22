#include "tensorrt_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <wrl/client.h>
#include <dxgi.h>

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

}