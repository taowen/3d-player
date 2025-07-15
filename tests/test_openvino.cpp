#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "openvino/openvino.hpp"

TEST_CASE("OpenVINO NPU ONNX inference", "[openvino]") {
    SECTION("NPU inference with simple ONNX model") {
        try {
            // 初始化 OpenVINO Core
            ov::Core core;
            
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
                std::cout << "NPU device not found, skipping NPU inference test" << std::endl;
                WARN("NPU device not available - test skipped");
                return;
            }
            
            // 加载简单的 ONNX 模型（两个数相加）
            std::string model_path = "test_data/simple_add.onnx";
            
            // 编译模型到 NPU 设备
            ov::CompiledModel compiled_model = core.compile_model(model_path, "NPU");
            
            // 创建推理请求
            ov::InferRequest infer_request = compiled_model.create_infer_request();
            
            // 获取输入输出信息
            auto inputs = compiled_model.inputs();
            auto outputs = compiled_model.outputs();
            
            // 验证模型有两个输入和一个输出
            REQUIRE(inputs.size() == 2);
            REQUIRE(outputs.size() == 1);
            
            auto input_a = inputs[0];  // A
            auto input_b = inputs[1];  // B
            auto output = outputs[0];  // C
            
            std::cout << "Input A shape: ";
            for (auto dim : input_a.get_shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Input B shape: ";
            for (auto dim : input_b.get_shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Output shape: ";
            for (auto dim : output.get_shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            
            // 创建输入数据
            std::vector<float> input_data_a = {1.0f, 2.0f};
            std::vector<float> input_data_b = {3.0f, 4.0f};
            
            // 创建输入张量
            ov::Tensor input_tensor_a(input_a.get_element_type(), input_a.get_shape(), input_data_a.data());
            ov::Tensor input_tensor_b(input_b.get_element_type(), input_b.get_shape(), input_data_b.data());
            
            // 设置输入 - 通过索引设置
            infer_request.set_input_tensor(0, input_tensor_a);  // 输入 A
            infer_request.set_input_tensor(1, input_tensor_b);  // 输入 B
            
            // 执行推理
            infer_request.infer();
            
            // 获取输出 - 通过索引获取
            ov::Tensor output_tensor = infer_request.get_output_tensor(0);  // 输出 C
            const float* output_data = output_tensor.data<const float>();
            
            // 验证结果 - 应该是 [4.0, 6.0]
            std::cout << "Inference results: ";
            for (size_t i = 0; i < output_tensor.get_size(); ++i) {
                std::cout << output_data[i] << " ";
            }
            std::cout << std::endl;
            
            // 验证结果是否正确
            REQUIRE(std::abs(output_data[0] - 4.0f) < 1e-5);
            REQUIRE(std::abs(output_data[1] - 6.0f) < 1e-5);
            
            std::cout << "NPU inference completed successfully!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Exception caught: " << e.what() << std::endl;
            FAIL("OpenVINO NPU inference failed: " + std::string(e.what()));
        }
    }
} 