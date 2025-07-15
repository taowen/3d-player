#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include <iostream>
#include <string>
#include <cstring>

#include "openvino/openvino.hpp"

TEST_CASE("OpenVINO basic functionality", "[openvino]") {
    SECTION("OpenVINO version information") {
        // 获取 OpenVINO 版本信息
        ov::Version version = ov::get_openvino_version();
        
        // 验证版本信息不为空
        REQUIRE(version.buildNumber != nullptr);
        REQUIRE(strlen(version.buildNumber) > 0);
        std::cout << "OpenVINO Version: " << version.buildNumber << std::endl;
        std::cout << "Description: " << version.description << std::endl;
    }
    
    SECTION("OpenVINO Core initialization") {
        // 初始化 OpenVINO Core
        ov::Core core;
        REQUIRE_NOTHROW([&]() { ov::Core test_core; }());
        
        // 获取可用设备列表
        std::vector<std::string> available_devices;
        REQUIRE_NOTHROW(available_devices = core.get_available_devices());
        
        // 验证设备列表不为空（至少应该有 CPU）
        REQUIRE_FALSE(available_devices.empty());
        
        std::cout << "Available devices: ";
        for (const auto& device : available_devices) {
            std::cout << device << " ";
        }
        std::cout << std::endl;
        
        // 验证 CPU 设备存在
        bool cpu_found = false;
        for (const auto& device : available_devices) {
            if (device == "CPU") {
                cpu_found = true;
                break;
            }
        }
        REQUIRE(cpu_found);
    }
    
    SECTION("OpenVINO Core device properties") {
        ov::Core core;
        
        // 获取 CPU 设备的支持属性
        std::vector<ov::PropertyName> supported_properties;
        REQUIRE_NOTHROW(supported_properties = core.get_property("CPU", ov::supported_properties));
        
        // 验证支持属性列表不为空
        REQUIRE_FALSE(supported_properties.empty());
        
        std::cout << "CPU supported properties count: " << supported_properties.size() << std::endl;
    }
} 