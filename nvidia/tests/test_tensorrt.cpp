#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

TEST_CASE("TensorRT ONNX inference", "[tensorrt][stereo_depth][test_tensorrt.cpp]") {
    SECTION("GPU inference with stereo depth ONNX model") {
        try {
            // 初始化 TensorRT Logger
            Logger logger;
            
            // 创建 TensorRT Runtime 和 Builder
            auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
            auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
            
            if (!runtime || !builder) {
                FAIL("Failed to create TensorRT runtime or builder");
                return;
            }
            
            std::cout << "TensorRT runtime and builder created successfully" << std::endl;
            
            // 检查 ONNX 模型文件是否存在
            std::string model_path = "../stereo_module_half_sbs.onnx";
            std::ifstream file(model_path, std::ios::binary);
            if (!file.is_open()) {
                std::cout << "ONNX model not found: " << model_path << std::endl;
                WARN("ONNX model file not available - TensorRT test skipped");
                return;
            }
            
            std::cout << "Loading ONNX model: " << model_path << std::endl;
            
            // 先尝试加载缓存的引擎
            std::string trt_cache_path = "../stereo_module_half_sbs_fp16.trt";
            std::unique_ptr<ICudaEngine> engine = nullptr;
            
            // 检查缓存文件是否存在
            std::ifstream cache_file(trt_cache_path, std::ios::binary);
            if (cache_file.is_open()) {
                std::cout << "Found cached TensorRT engine: " << trt_cache_path << std::endl;
                
                // 获取文件大小
                cache_file.seekg(0, std::ios::end);
                size_t cache_size = cache_file.tellg();
                cache_file.seekg(0, std::ios::beg);
                
                if (cache_size > 0) {
                    // 读取缓存数据
                    std::vector<char> cache_data(cache_size);
                    cache_file.read(cache_data.data(), cache_size);
                    cache_file.close();
                    
                    // 反序列化引擎
                    engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(
                        cache_data.data(), cache_size));
                    
                    if (engine) {
                        std::cout << "Successfully loaded cached TensorRT engine!" << std::endl;
                    } else {
                        std::cout << "Failed to deserialize cached engine, will rebuild..." << std::endl;
                    }
                } else {
                    cache_file.close();
                    std::cout << "Cache file is empty, will rebuild..." << std::endl;
                }
            } else {
                std::cout << "No cached engine found, will build from ONNX..." << std::endl;
            }
            
            // 如果没有成功加载缓存引擎，则从ONNX构建
            if (!engine) {
                std::cout << "Building TensorRT engine from ONNX model..." << std::endl;
                
                // 创建网络定义
                const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
                auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicit_batch));
            
                // 创建 ONNX 解析器
                auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
                
                if (!parser) {
                    FAIL("Failed to create ONNX parser");
                    return;
                }
            
                // 解析 ONNX 模型
                bool parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
                if (!parsed) {
                    std::cout << "Failed to parse ONNX model. Errors:" << std::endl;
                    for (int i = 0; i < parser->getNbErrors(); i++) {
                        std::cout << "  " << parser->getError(i)->desc() << std::endl;
                    }
                    FAIL("ONNX model parsing failed");
                    return;
                }
                
                std::cout << "ONNX model parsed successfully" << std::endl;
                
                // 获取输入输出信息
                auto num_inputs = network->getNbInputs();
                auto num_outputs = network->getNbOutputs();
                
                std::cout << "Model inputs: " << num_inputs << ", outputs: " << num_outputs << std::endl;
                
                REQUIRE(num_inputs >= 1);
                REQUIRE(num_outputs >= 1);
                
                auto input = network->getInput(0);
                auto output = network->getOutput(0);
                
                auto input_dims = input->getDimensions();
                auto output_dims = output->getDimensions();
                
                std::cout << "Input shape: ";
                for (int i = 0; i < input_dims.nbDims; i++) {
                    std::cout << input_dims.d[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "Output shape: ";
                for (int i = 0; i < output_dims.nbDims; i++) {
                    std::cout << output_dims.d[i] << " ";
                }
                std::cout << std::endl;
                
                // 检查是否有动态维度
                bool has_dynamic_shapes = false;
                for (int i = 0; i < input_dims.nbDims; i++) {
                    if (input_dims.d[i] == -1) {
                        has_dynamic_shapes = true;
                        break;
                    }
                }
                
                std::cout << "Has dynamic shapes: " << (has_dynamic_shapes ? "Yes" : "No") << std::endl;
                
                // 获取输入张量名称
                const char* input_name = input->getName();
                
                // 创建配置
                auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
                if (!config) {
                    FAIL("Failed to create builder config");
                    return;
                }
                
                // 只有在有动态形状时才设置优化配置文件
                if (has_dynamic_shapes) {
                    std::cout << "Setting up optimization profile for dynamic shapes..." << std::endl;
                    
                    auto profile = builder->createOptimizationProfile();
                    if (!profile) {
                        FAIL("Failed to create optimization profile");
                        return;
                    }
                    
                    std::cout << "Input tensor: " << input_name << " with dimensions: ";
                    for (int i = 0; i < input_dims.nbDims; i++) {
                        std::cout << input_dims.d[i] << " ";
                    }
                    std::cout << std::endl;
                    
                    // 使用与 spike 参考实现相同的范围
                    std::cout << "Setting optimization profile:" << std::endl;
                    std::cout << "  Min: 1x4x512x512" << std::endl;
                    std::cout << "  Opt: 1x4x1080x1920" << std::endl; 
                    std::cout << "  Max: 1x4x2160x4096" << std::endl;
                    
                    // 使用与 spike/src/infer_iw3.cpp 相同的配置
                    profile->setDimensions(input_name, OptProfileSelector::kMIN, nvinfer1::Dims4{1, 4, 512, 512});
                    profile->setDimensions(input_name, OptProfileSelector::kOPT, nvinfer1::Dims4{1, 4, 1080, 1920});
                    profile->setDimensions(input_name, OptProfileSelector::kMAX, nvinfer1::Dims4{1, 4, 2160, 4096});
                    
                    // 验证配置文件有效性（根据 deepwiki 建议）
                    bool is_valid = profile->isValid();
                    std::cout << "Profile validation result: " << (is_valid ? "Valid" : "Invalid") << std::endl;
                    if (!is_valid) {
                        std::cerr << "Invalid optimization profile" << std::endl;
                        FAIL("Invalid optimization profile");
                        return;
                    }
                    
                    // 添加到构建配置 (直接调用，和 spike 实现一致)
                    int profile_index = config->addOptimizationProfile(profile);
                    if (profile_index < 0) {
                        std::cerr << "Failed to add optimization profile, index: " << profile_index << std::endl;
                        FAIL("Failed to add optimization profile");
                        return;
                    }
                    std::cout << "Optimization profile added to config with index: " << profile_index << std::endl;
                    
                    std::cout << "Optimization profile configured successfully!" << std::endl;
                } else {
                    std::cout << "No dynamic shapes detected, skipping optimization profile setup" << std::endl;
                }
                
                // 然后启用 FP16（参考 infer_iw3.cpp 顺序）
                if (builder->platformHasFastFp16()) {
                    std::cout << "GPU supports FP16, enabling FP16 mode" << std::endl;
                    config->setFlag(BuilderFlag::kFP16);
                }
                
                // 最后设置内存池大小 (1GB)
                config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);
                
                // 构建TensorRT引擎（使用新版API）
                std::cout << "Building TensorRT engine (this may take several minutes)..." << std::endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                
                // 使用 buildSerializedNetwork 代替 buildEngineWithConfig
                auto serializedEngine = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
                
                if (!serializedEngine) {
                    FAIL("Failed to build TensorRT engine");
                    return;
                }
                
                // 保存缓存
                std::ofstream cache_out(trt_cache_path, std::ios::binary);
                if (cache_out.is_open()) {
                    cache_out.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
                    cache_out.close();
                    std::cout << "Saved TensorRT engine cache to: " << trt_cache_path << std::endl;
                } else {
                    std::cout << "Failed to save TensorRT engine cache" << std::endl;
                }
                
                // 反序列化引擎
                engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(
                    serializedEngine->data(), serializedEngine->size()));
                
                if (!engine) {
                    FAIL("Failed to deserialize TensorRT engine");
                    return;
                }
                
                std::cout << "TensorRT engine built successfully in " << duration.count() << " seconds" << std::endl;
            }
            
            // 创建执行上下文
            auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
            if (!context) {
                FAIL("Failed to create execution context");
                return;
            }
            
            // 设置运行时输入形状（使用测试尺寸 1080p）
            Dims runtime_dims = {4, {1, 4, 1080, 1920}};  // 使用1080p作为测试尺寸
            const char* input_tensor_name = engine->getIOTensorName(0);
            if (!context->setInputShape(input_tensor_name, runtime_dims)) {
                FAIL("Failed to set runtime input shape");
                return;
            }
            
            std::cout << "Set runtime input shape to: " 
                      << runtime_dims.d[0] << "x" << runtime_dims.d[1] << "x" << runtime_dims.d[2] << "x" << runtime_dims.d[3] << std::endl;
            
            // 验证引擎信息
            auto num_bindings = engine->getNbIOTensors();
            std::cout << "Engine I/O tensors: " << num_bindings << std::endl;
            
            REQUIRE(num_bindings >= 2); // 至少一个输入和一个输出
            
            // 显示所有张量信息
            for (int i = 0; i < num_bindings; i++) {
                auto tensor_name = engine->getIOTensorName(i);
                auto tensor_mode = engine->getTensorIOMode(tensor_name);
                auto tensor_shape = context->getTensorShape(tensor_name);  // 使用context获取运行时形状
                
                std::cout << "Tensor " << i << ": " << tensor_name 
                          << " (" << (tensor_mode == TensorIOMode::kINPUT ? "input" : "output") << ") ";
                std::cout << "shape: ";
                for (int j = 0; j < tensor_shape.nbDims; j++) {
                    std::cout << tensor_shape.d[j] << " ";
                }
                std::cout << std::endl;
            }
            
            // 现在执行实际的推理测试
            std::cout << "\n--- 开始推理测试 ---" << std::endl;
            
            // 获取输入输出张量的运行时形状和大小
            auto output_tensor_name = engine->getIOTensorName(1);
            
            auto input_shape = context->getTensorShape(input_tensor_name);
            auto output_shape = context->getTensorShape(output_tensor_name);
            
            // 计算数据大小
            size_t input_elements = 1;
            for (int i = 0; i < input_shape.nbDims; ++i) {
                input_elements *= input_shape.d[i];
            }
            size_t input_size_bytes = input_elements * sizeof(float);
            
            size_t output_elements = 1;
            for (int i = 0; i < output_shape.nbDims; ++i) {
                output_elements *= output_shape.d[i];
            }
            size_t output_size_bytes = output_elements * sizeof(float);
            
            std::cout << "Input size: " << input_elements << " elements (" << input_size_bytes << " bytes)" << std::endl;
            std::cout << "Output size: " << output_elements << " elements (" << output_size_bytes << " bytes)" << std::endl;
            
            // 分配主机内存
            std::vector<float> host_input(input_elements);
            std::vector<float> host_output(output_elements);
            
            // 生成测试数据 (RGBA格式，值范围 [0.0, 1.0])
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);
            
            for (size_t i = 0; i < input_elements; ++i) {
                host_input[i] = dis(gen);
            }
            
            std::cout << "Generated random test data with " << input_elements << " values" << std::endl;
            
            // 分配设备内存
            void* device_input = nullptr;
            void* device_output = nullptr;
            
            cudaError_t cuda_err = cudaMalloc(&device_input, input_size_bytes);
            REQUIRE(cuda_err == cudaSuccess);
            
            cuda_err = cudaMalloc(&device_output, output_size_bytes);
            REQUIRE(cuda_err == cudaSuccess);
            
            std::cout << "Allocated device memory for input and output" << std::endl;
            
            // 拷贝输入数据到设备
            cuda_err = cudaMemcpy(device_input, host_input.data(), input_size_bytes, cudaMemcpyHostToDevice);
            REQUIRE(cuda_err == cudaSuccess);
            
            // 设置输入输出绑定
            void* bindings[2];
            bindings[0] = device_input;
            bindings[1] = device_output;
            
            // 执行推理
            std::cout << "Executing inference..." << std::endl;
            auto inference_start = std::chrono::high_resolution_clock::now();
            
            bool inference_success = context->executeV2(bindings);
            
            auto inference_end = std::chrono::high_resolution_clock::now();
            auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start);
            
            REQUIRE(inference_success);
            std::cout << "Inference completed successfully in " << inference_duration.count() << " ms" << std::endl;
            
            // 拷贝输出数据回主机
            cuda_err = cudaMemcpy(host_output.data(), device_output, output_size_bytes, cudaMemcpyDeviceToHost);
            REQUIRE(cuda_err == cudaSuccess);
            
            std::cout << "Copied output data back to host" << std::endl;
            
            // 验证输出数据
            bool has_valid_output = false;
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            
            for (size_t i = 0; i < output_elements; ++i) {
                float val = host_output[i];
                if (!std::isnan(val) && !std::isinf(val)) {
                    has_valid_output = true;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }
            
            REQUIRE(has_valid_output);
            std::cout << "Output validation passed:" << std::endl;
            std::cout << "  Value range: [" << min_val << ", " << max_val << "]" << std::endl;
            std::cout << "  No NaN or Inf values detected" << std::endl;
            
            // 显示一些输出样本
            std::cout << "Output samples (first 10 values): ";
            for (int i = 0; i < std::min(10, (int)output_elements); ++i) {
                std::cout << host_output[i] << " ";
            }
            std::cout << std::endl;
            
            // 清理 CUDA 内存
            cudaFree(device_input);
            cudaFree(device_output);
            
            std::cout << "--- 推理测试完成 ---" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Exception caught: " << e.what() << std::endl;
            FAIL("TensorRT ONNX inference failed: " + std::string(e.what()));
        }
    }
}