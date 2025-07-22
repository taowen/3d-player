#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

// D3D11 headers
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

// CUDA headers
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

using Microsoft::WRL::ComPtr;

TEST_CASE("D3D11 CUDA Interop", "[d3d11][cuda][interop][test_d3d11_cuda_interop.cpp]") {
    SECTION("Texture to CUDA memory mapping") {
        
        // 1. 创建 D3D11 设备 - 使用和 CUDA 兼容的适配器
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        D3D_FEATURE_LEVEL feature_level;
        
        // 枚举所有适配器，找到支持 CUDA 的
        ComPtr<IDXGIFactory> factory;
        HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), &factory);
        REQUIRE(SUCCEEDED(hr));
        
        ComPtr<IDXGIAdapter> cuda_compatible_adapter;
        bool found_cuda_adapter = false;
        
        // 枚举所有适配器
        for (UINT adapter_index = 0; ; ++adapter_index) {
            ComPtr<IDXGIAdapter> adapter;
            if (FAILED(factory->EnumAdapters(adapter_index, &adapter))) {
                break; // 没有更多适配器
            }
            
            // 检查这个适配器是否支持 CUDA
            int cuda_device = -1;
            cudaError_t cuda_err = cudaD3D11GetDevice(&cuda_device, adapter.Get());
            if (cuda_err == cudaSuccess) {
                cuda_compatible_adapter = adapter;
                found_cuda_adapter = true;
                std::cout << "Found CUDA-compatible adapter at index " << adapter_index << ", CUDA device " << cuda_device << std::endl;
                break;
            } else {
                std::cout << "Adapter " << adapter_index << " is not CUDA-compatible: " << cudaGetErrorString(cuda_err) << std::endl;
            }
        }
        
        if (!found_cuda_adapter) {
            std::cout << "No CUDA-compatible adapter found, skipping interop test" << std::endl;
            REQUIRE(found_cuda_adapter);
        }
        
        hr = D3D11CreateDevice(
            cuda_compatible_adapter.Get(),  // 使用 CUDA 兼容的适配器
            D3D_DRIVER_TYPE_UNKNOWN,  // 必须是 UNKNOWN 当指定适配器时
            nullptr,
            0,
            nullptr, 0,
            D3D11_SDK_VERSION,
            &device,
            &feature_level,
            &context
        );
        
        REQUIRE(SUCCEEDED(hr));
        REQUIRE(device != nullptr);
        REQUIRE(context != nullptr);
        
        std::cout << "D3D11 device created successfully" << std::endl;
        
        // 2. 先检查 CUDA 设备数量
        int device_count;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        std::cout << "CUDA device count: " << device_count << ", error: " << cuda_err << std::endl;
        
        if (cuda_err != cudaSuccess) {
            std::cout << "CUDA initialization failed: " << cudaGetErrorString(cuda_err) << std::endl;
            REQUIRE(cuda_err == cudaSuccess);
        }
        
        // 3. 尝试设置 CUDA 设备
        cuda_err = cudaSetDevice(0);
        if (cuda_err != cudaSuccess) {
            std::cout << "Failed to set CUDA device 0: " << cudaGetErrorString(cuda_err) << std::endl;
            REQUIRE(cuda_err == cudaSuccess);
        }
        std::cout << "CUDA device 0 set successfully" << std::endl;
        
        // 4. 获取对应的 CUDA 设备（由于我们已经使用 CUDA 兼容的适配器，这应该会成功）
        ComPtr<IDXGIDevice> dxgi_device;
        hr = device->QueryInterface(__uuidof(IDXGIDevice), &dxgi_device);
        REQUIRE(SUCCEEDED(hr));
        
        ComPtr<IDXGIAdapter> dxgi_adapter;
        hr = dxgi_device->GetAdapter(&dxgi_adapter);
        REQUIRE(SUCCEEDED(hr));
        
        int cuda_device = -1;
        cuda_err = cudaD3D11GetDevice(&cuda_device, dxgi_adapter.Get());
        REQUIRE(cuda_err == cudaSuccess);
        
        cuda_err = cudaSetDevice(cuda_device);
        REQUIRE(cuda_err == cudaSuccess);
        std::cout << "Successfully set CUDA device to " << cuda_device << std::endl;
        
        std::cout << "CUDA-D3D11 interop initialized" << std::endl;
        
        // 3. 创建一个测试纹理 (1920x1080 RGBA32F)
        const int width = 1920;
        const int height = 1080;
        
        D3D11_TEXTURE2D_DESC texture_desc = {};
        texture_desc.Width = width;
        texture_desc.Height = height;
        texture_desc.MipLevels = 1;
        texture_desc.ArraySize = 1;
        texture_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        texture_desc.SampleDesc.Count = 1;
        texture_desc.Usage = D3D11_USAGE_DEFAULT;
        texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        texture_desc.CPUAccessFlags = 0;
        texture_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;  // CUDA interop 需要共享标志
        
        ComPtr<ID3D11Texture2D> d3d_texture;
        hr = device->CreateTexture2D(&texture_desc, nullptr, &d3d_texture);
        REQUIRE(SUCCEEDED(hr));
        
        std::cout << "Created D3D11 texture: " << width << "x" << height << " RGBA32F" << std::endl;
        
        // 4. 注册 D3D11 纹理到 CUDA
        cudaGraphicsResource* cuda_resource = nullptr;
        cuda_err = cudaGraphicsD3D11RegisterResource(
            &cuda_resource, 
            d3d_texture.Get(), 
            cudaGraphicsRegisterFlagsNone  // 读写权限
        );
        REQUIRE(cuda_err == cudaSuccess);
        REQUIRE(cuda_resource != nullptr);
        
        std::cout << "D3D11 texture registered to CUDA graphics resource" << std::endl;
        
        // 5. 映射 CUDA 资源
        cuda_err = cudaGraphicsMapResources(1, &cuda_resource, 0);
        REQUIRE(cuda_err == cudaSuccess);
        
        std::cout << "CUDA graphics resource mapped" << std::endl;
        
        // 6. 获取映射后的 CUDA 数组
        cudaArray_t cuda_array = nullptr;
        cuda_err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
        REQUIRE(cuda_err == cudaSuccess);
        REQUIRE(cuda_array != nullptr);
        
        std::cout << "Got mapped CUDA array from graphics resource" << std::endl;
        
        // 7. 创建 CUDA Surface Object 用于写入
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;
        
        cudaSurfaceObject_t surf_obj = 0;
        cuda_err = cudaCreateSurfaceObject(&surf_obj, &res_desc);
        REQUIRE(cuda_err == cudaSuccess);
        REQUIRE(surf_obj != 0);
        
        std::cout << "Created CUDA surface object" << std::endl;
        
        // 8. 分配临时的 CUDA 线性内存用于数据生成和验证
        size_t pixel_size = 4 * sizeof(float);  // RGBA32F
        size_t image_size = width * height * pixel_size;
        
        void* cuda_linear_memory = nullptr;
        cuda_err = cudaMalloc(&cuda_linear_memory, image_size);
        REQUIRE(cuda_err == cudaSuccess);
        
        std::cout << "Allocated CUDA linear memory: " << image_size << " bytes" << std::endl;
        
        // 9. 在设备上生成测试数据 (简单的渐变图案)
        // 这里需要一个简单的 CUDA kernel
        // 先用 cudaMemset 简单测试
        cuda_err = cudaMemset(cuda_linear_memory, 0xFF, image_size);
        REQUIRE(cuda_err == cudaSuccess);
        
        std::cout << "Filled CUDA linear memory with test pattern" << std::endl;
        
        // 10. 从线性内存拷贝到 Surface (纹理)
        cuda_err = cudaMemcpy2DToArray(
            cuda_array,
            0, 0,  // offset
            cuda_linear_memory,
            width * pixel_size,  // src pitch
            width * pixel_size,  // width in bytes
            height,              // height
            cudaMemcpyDeviceToDevice
        );
        REQUIRE(cuda_err == cudaSuccess);
        
        std::cout << "Copied data from CUDA linear memory to texture array" << std::endl;
        
        // 11. 测试反向操作：从纹理读回线性内存
        void* cuda_readback_memory = nullptr;
        cuda_err = cudaMalloc(&cuda_readback_memory, image_size);
        REQUIRE(cuda_err == cudaSuccess);
        
        cuda_err = cudaMemcpy2DFromArray(
            cuda_readback_memory,
            width * pixel_size,  // dst pitch
            cuda_array,
            0, 0,  // offset
            width * pixel_size,  // width in bytes
            height,              // height
            cudaMemcpyDeviceToDevice
        );
        REQUIRE(cuda_err == cudaSuccess);
        
        std::cout << "Read back data from texture array to CUDA linear memory" << std::endl;
        
        // 12. 验证数据一致性 (比较前几个字节)
        std::vector<uint8_t> host_original(256);
        std::vector<uint8_t> host_readback(256);
        
        cuda_err = cudaMemcpy(host_original.data(), cuda_linear_memory, 256, cudaMemcpyDeviceToHost);
        REQUIRE(cuda_err == cudaSuccess);
        
        cuda_err = cudaMemcpy(host_readback.data(), cuda_readback_memory, 256, cudaMemcpyDeviceToHost);
        REQUIRE(cuda_err == cudaSuccess);
        
        bool data_matches = true;
        for (int i = 0; i < 256; ++i) {
            if (host_original[i] != host_readback[i]) {
                data_matches = false;
                break;
            }
        }
        
        REQUIRE(data_matches);
        std::cout << "Data integrity verification passed!" << std::endl;
        
        // 13. 清理 Surface Object
        cuda_err = cudaDestroySurfaceObject(surf_obj);
        REQUIRE(cuda_err == cudaSuccess);
        
        // 14. 解除映射
        cuda_err = cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        REQUIRE(cuda_err == cudaSuccess);
        
        std::cout << "Unmapped CUDA graphics resource" << std::endl;
        
        // 15. 注销资源
        cuda_err = cudaGraphicsUnregisterResource(cuda_resource);
        REQUIRE(cuda_err == cudaSuccess);
        
        std::cout << "Unregistered CUDA graphics resource" << std::endl;
        
        // 16. 清理 CUDA 内存
        cudaFree(cuda_linear_memory);
        cudaFree(cuda_readback_memory);
        
        std::cout << "D3D11-CUDA interop test completed successfully!" << std::endl;
    }
}