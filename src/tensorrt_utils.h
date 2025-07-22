#pragma once

#include <NvInfer.h>
#include <string>

namespace TensorRTUtils {
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    bool initializeCudaDevice(void* d3d11_device, int& cuda_device);
    
    nvinfer1::IRuntime* createTensorRTRuntime(nvinfer1::ILogger& logger);
    
    nvinfer1::ICudaEngine* loadEngineFromCache(nvinfer1::IRuntime* runtime, const std::string& cache_path);
}