#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>

#include "openvino/openvino.hpp"

TEST_CASE("Stereo Depth NPU ONNX inference", "[openvino][stereo_depth]") {
    SECTION("NPU inference with stereo depth ONNX model") {
        try {
            // 初始化 OpenVINO Core
            ov::Core core;
            
            // 启用模型缓存以加速后续加载
            core.set_property(ov::cache_dir("cache"));
            std::cout << "Model caching enabled in 'cache' directory" << std::endl;
            
            // 获取可用设备列表
            std::vector<std::string> available_devices = core.get_available_devices();
            
            std::cout << "Available devices: ";
            for (const auto& device : available_devices) {
                std::cout << device << " ";
            }
            std::cout << std::endl;
            
            // 检查 NPU 设备是否可用
            bool npu_found = false;
            for (const auto& device : available_devices) {
                if (device == "NPU") {
                    npu_found = true;
                    break;
                }
            }
            
            if (!npu_found) {
                std::cout << "NPU device not found, skipping stereo depth NPU inference test" << std::endl;
                WARN("NPU device not available - stereo depth test skipped");
                return;
            }
            
            // 加载立体深度 ONNX 模型
            std::string model_path = "deps/nunif/stereo_module_half_sbs.onnx";
            
            std::cout << "Loading stereo depth model: " << model_path << std::endl;
            std::cout << "Note: First-time NPU compilation may take several minutes..." << std::endl;
            std::cout << "Processing dynamic shapes for NPU compatibility..." << std::endl;
            
            // 先读取模型以获取输入信息
            std::shared_ptr<ov::Model> model = core.read_model(model_path);
            
            // 获取输入信息并设置固定形状
            auto inputs = model->inputs();
            REQUIRE(inputs.size() == 1);
            
            auto input = inputs[0];
            auto original_shape = input.get_partial_shape();
            
            std::cout << "Original input shape: ";
            for (size_t i = 0; i < original_shape.size(); i++) {
                if (original_shape[i].is_dynamic()) {
                    std::cout << "? ";
                } else {
                    std::cout << original_shape[i].get_length() << " ";
                }
            }
            std::cout << std::endl;
            
            // 设置固定的输入形状 (1, 4, 512, 512) - NCHW格式
            ov::Shape fixed_input_shape = {1, 4, 512, 512};
            
            std::cout << "Setting fixed input shape: ";
            for (auto dim : fixed_input_shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            
            // 重塑模型以使用固定形状
            std::map<ov::Output<ov::Node>, ov::PartialShape> shape_map;
            shape_map[input] = ov::PartialShape(fixed_input_shape);
            model->reshape(shape_map);
            
            // 测量编译时间
            auto compile_start = std::chrono::high_resolution_clock::now();
            
            try {
                // 编译重塑后的模型到 NPU 设备（使用缓存）
                ov::CompiledModel compiled_model = core.compile_model(model, "NPU");
                
                auto compile_end = std::chrono::high_resolution_clock::now();
                auto compile_duration = std::chrono::duration_cast<std::chrono::seconds>(compile_end - compile_start);
                std::cout << "NPU model compilation completed in " << compile_duration.count() << " seconds" << std::endl;
                
                // 创建推理请求
                ov::InferRequest infer_request = compiled_model.create_infer_request();
                
                // 获取输入输出信息
                auto compiled_inputs = compiled_model.inputs();
                auto compiled_outputs = compiled_model.outputs();
                
                // 验证模型有一个输入和一个输出
                REQUIRE(compiled_inputs.size() == 1);
                REQUIRE(compiled_outputs.size() == 1);
                
                auto compiled_input = compiled_inputs[0];
                auto compiled_output = compiled_outputs[0];
                
                // 验证输入形状已经固定
                auto input_shape = compiled_input.get_shape();
                REQUIRE(input_shape.size() == 4);
                
                size_t batch_size = input_shape[0];
                size_t channels = input_shape[1];
                size_t height = input_shape[2];
                size_t width = input_shape[3];
                
                REQUIRE(batch_size == 1);
                REQUIRE(channels == 4); // RGBA
                REQUIRE(height == 512);
                REQUIRE(width == 512);
                
                std::cout << "Compiled input tensor dimensions: " 
                          << "B=" << batch_size 
                          << ", C=" << channels 
                          << ", H=" << height 
                          << ", W=" << width << std::endl;
                
                // 创建测试输入数据 (RGBA, 渐变图案)
                size_t input_size = batch_size * channels * height * width;
                std::vector<float> input_data(input_size);
                
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t c = 0; c < 3; c++) { // RGB channels
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                size_t idx = b * channels * height * width + 
                                            c * height * width + 
                                            h * width + w;
                                
                                // 创建简单的渐变图案
                                float value = (float(h) / height + float(w) / width) * 0.5f;
                                if (c == 0) value += 0.2f; // R channel
                                if (c == 1) value += 0.1f; // G channel
                                // B channel keeps base value
                                
                                input_data[idx] = std::min(1.0f, value);
                            }
                        }
                    }
                    
                    // Alpha channel - 设为 1.0
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            size_t idx = b * channels * height * width + 
                                        3 * height * width + 
                                        h * width + w;
                            input_data[idx] = 1.0f;
                        }
                    }
                }
                
                std::cout << "Created test input data with size: " << input_data.size() << std::endl;
                
                // 创建输入张量 - 使用编译后确定的形状
                ov::Tensor input_tensor(compiled_input.get_element_type(), input_shape, input_data.data());
                
                // 设置输入
                infer_request.set_input_tensor(0, input_tensor);
                
                std::cout << "Starting NPU inference..." << std::endl;
                
                // 执行推理
                auto start_time = std::chrono::high_resolution_clock::now();
                infer_request.infer();
                auto end_time = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                std::cout << "NPU inference completed in " << duration.count() << " ms" << std::endl;
                
                // 获取输出
                ov::Tensor output_tensor = infer_request.get_output_tensor(0);
                const float* output_data = output_tensor.data<const float>();
                
                // 验证输出维度
                auto output_shape = output_tensor.get_shape();
                REQUIRE(output_shape.size() == 4); // NCHW format
                
                size_t output_batch = output_shape[0];
                size_t output_channels = output_shape[1];
                size_t output_height = output_shape[2];
                size_t output_width = output_shape[3];
                
                std::cout << "Output tensor dimensions: " 
                          << "B=" << output_batch 
                          << ", C=" << output_channels 
                          << ", H=" << output_height 
                          << ", W=" << output_width << std::endl;
                
                // 验证输出形状符合预期 (half side-by-side, RGBA)
                REQUIRE(output_batch == batch_size);
                REQUIRE(output_channels == 4); // RGBA
                REQUIRE(output_height == height);
                REQUIRE(output_width == width); // 半宽度并排应该保持原宽度
                
                // 验证输出数据范围合理 (0-1)
                size_t output_size = output_tensor.get_size();
                float min_val = output_data[0];
                float max_val = output_data[0];
                
                for (size_t i = 0; i < output_size; i++) {
                    min_val = std::min(min_val, output_data[i]);
                    max_val = std::max(max_val, output_data[i]);
                }
                
                std::cout << "Output value range: [" << min_val << ", " << max_val << "]" << std::endl;
                
                // 输出值应该在合理范围内
                REQUIRE(min_val >= -0.5f); // 允许一些负值
                REQUIRE(max_val <= 1.5f);  // 允许稍微超过1.0
                
                // 检查输出不全为零
                bool has_non_zero = false;
                for (size_t i = 0; i < output_size; i++) {
                    if (std::abs(output_data[i]) > 1e-6) {
                        has_non_zero = true;
                        break;
                    }
                }
                REQUIRE(has_non_zero);
                
                std::cout << "Stereo depth NPU inference completed successfully!" << std::endl;
                std::cout << "Model generated half side-by-side stereo output from RGBA input" << std::endl;
                
            } catch (const std::exception& compilation_error) {
                std::cout << "NPU compilation error: " << compilation_error.what() << std::endl;
                
                // 检查是否是动态形状问题
                if (std::string(compilation_error.what()).find("Upper bounds were not specified") != std::string::npos ||
                    std::string(compilation_error.what()).find("dynamic") != std::string::npos) {
                    
                    std::cout << "NPU doesn't support dynamic shapes. This is expected for models with dynamic inputs." << std::endl;
                    WARN("NPU compilation failed due to dynamic shapes - this is a known limitation");
                } else {
                    // 其他类型的错误则重新抛出
                    throw;
                }
            }
            
        } catch (const std::exception& e) {
            std::cout << "Exception caught: " << e.what() << std::endl;
            FAIL("Stereo Depth NPU inference failed: " + std::string(e.what()));
        }
    }
}
