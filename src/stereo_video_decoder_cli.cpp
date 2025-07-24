#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include "stereo_video_decoder.h"

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

// 保存CUDA浮点缓冲区为BMP文件
void saveCudaFloatBufferToBMP(void* cuda_buffer, int width, int height, const std::string& filename) {
    if (!cuda_buffer) return;
    
    // 分配主机内存
    size_t total_pixels = width * height * 4; // RGBA
    std::vector<float> host_data(total_pixels);
    
    // 从GPU复制到CPU
    cudaError_t cuda_err = cudaMemcpy(host_data.data(), cuda_buffer, 
                                      total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy CUDA buffer to host: " << cudaGetErrorString(cuda_err) << std::endl;
        return;
    }
    
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
        
        // BMP格式是从下到上存储的，所以需要翻转Y轴
        for (int y = height - 1; y >= 0; y--) {
            for (int x = 0; x < width; x++) {
                int pixel_index = (y * width + x) * 4; // RGBA
                float r = host_data[pixel_index + 0];
                float g = host_data[pixel_index + 1];
                float b = host_data[pixel_index + 2];
                
                // 将浮点数转换为0-255范围的整数
                uint8_t r_byte = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
                uint8_t g_byte = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
                uint8_t b_byte = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
                
                // BMP格式是BGR而不是RGB
                int row_index = x * 3;
                row_data[row_index + 0] = b_byte; // B
                row_data[row_index + 1] = g_byte; // G
                row_data[row_index + 2] = r_byte; // R
            }
            file.write(reinterpret_cast<const char*>(row_data.data()), row_size);
        }
        file.close();
        std::cout << "Saved BMP file: " << filename << std::endl;
    } else {
        std::cerr << "Failed to create BMP file: " << filename << std::endl;
    }
}

// 保存D3D11浮点纹理为BMP文件
void saveD3D11FloatTextureToBMP(ID3D11Device* device, ID3D11DeviceContext* context, 
                                ID3D11Texture2D* texture, const std::string& filename) {
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    // 创建staging纹理
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging_texture;
    HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture" << std::endl;
        return;
    }
    
    context->CopyResource(staging_texture.Get(), texture);
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture, HRESULT: " << std::hex << hr << std::endl;
        return;
    }
    
    // 验证映射的数据指针
    if (!mapped.pData) {
        std::cerr << "Mapped data pointer is null" << std::endl;
        return;
    }
    
    // 输出调试信息
    std::cout << "Texture format: " << desc.Format << std::endl;
    std::cout << "RowPitch: " << mapped.RowPitch << std::endl;
    //std::cout << "Expected row size: " << width * 4 * sizeof(float) << std::endl;
    
    uint32_t width = desc.Width;
    uint32_t height = desc.Height;
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
        
        // BMP格式是从下到上存储的，所以需要翻转Y轴
        for (int y = height - 1; y >= 0; y--) {
            // 检查单行数据大小是否超出RowPitch
            size_t row_data_size = width * 4 * sizeof(float);
            if (row_data_size > mapped.RowPitch) {
                std::cerr << "Row data size (" << row_data_size << ") exceeds RowPitch (" << mapped.RowPitch << ")" << std::endl;
                context->Unmap(staging_texture.Get(), 0);
                return;
            }
            
            size_t row_offset = y * mapped.RowPitch;
            
            const float* src_row = reinterpret_cast<const float*>(
                static_cast<const uint8_t*>(mapped.pData) + row_offset);
            
            for (int x = 0; x < static_cast<int>(width); x++) {
                // 假设格式是R32G32B32A32_FLOAT
                float r = src_row[x * 4 + 0];
                float g = src_row[x * 4 + 1];
                float b = src_row[x * 4 + 2];
                
                // 将浮点数转换为0-255范围的整数
                uint8_t r_byte = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
                uint8_t g_byte = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
                uint8_t b_byte = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
                
                // BMP格式是BGR而不是RGB
                int row_index = x * 3;
                row_data[row_index + 0] = b_byte; // B
                row_data[row_index + 1] = g_byte; // G
                row_data[row_index + 2] = r_byte; // R
            }
            file.write(reinterpret_cast<const char*>(row_data.data()), row_size);
        }
        file.close();
        std::cout << "Saved BMP file: " << filename << std::endl;
    } else {
        std::cerr << "Failed to create BMP file: " << filename << std::endl;
    }
    
    context->Unmap(staging_texture.Get(), 0);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <video_file> <frame_number>" << std::endl;
        std::cerr << "Example: " << argv[0] << " test_data/sample_hw.mkv 10" << std::endl;
        return 1;
    }
    
    std::string video_file = argv[1];
    int target_frame = std::stoi(argv[2]);
    
    if (target_frame < 0) {
        std::cerr << "Frame number must be non-negative" << std::endl;
        return 1;
    }
    
    // 初始化StereoVideoDecoder
    StereoVideoDecoder decoder;
    if (!decoder.open(video_file)) {
        std::cerr << "Failed to open video file: " << video_file << std::endl;
        return 1;
    }
    
    std::cout << "Video opened successfully. Resolution: " << decoder.getWidth() 
              << "x" << decoder.getHeight() << std::endl;
    
    // 跳到第n帧
    DecodedStereoFrame frame;
    int current_frame = 0;
    
    while (current_frame <= target_frame && !decoder.isEOF()) {
        if (!decoder.readNextFrame(frame)) {
            std::cerr << "Failed to read frame " << current_frame << std::endl;
            return 1;
        }
        
        if (current_frame == target_frame) {
            std::cout << "Processing frame " << target_frame << std::endl;
            
            // 保存输入帧（CUDA缓冲区）
            if (frame.input_frame && frame.input_frame->cuda_buffer && frame.input_frame->buffer_size > 0) {
                std::string input_filename = "frame_" + std::to_string(target_frame) + "_input.bmp";
                int input_width = decoder.getWidth();
                int input_height = decoder.getHeight();
                saveCudaFloatBufferToBMP(frame.input_frame->cuda_buffer, input_width, 
                                       input_height, input_filename);
            } else {
                std::cerr << "Warning: No input CUDA buffer available for frame " << target_frame << std::endl;
            }
            
            // 保存输出帧（CUDA缓冲区）
            if (frame.cuda_output_buffer && frame.output_width > 0 && frame.output_height > 0) {
                std::string output_filename = "frame_" + std::to_string(target_frame) + "_output.bmp";
                saveCudaFloatBufferToBMP(frame.cuda_output_buffer, frame.output_width, 
                                       frame.output_height, output_filename);
            } else {
                std::cerr << "Warning: No CUDA output buffer available for frame " << target_frame << std::endl;
            }
            
            break;
        }
        
        current_frame++;
    }
    
    if (current_frame < target_frame) {
        std::cerr << "Video has only " << current_frame << " frames, cannot reach frame " 
                  << target_frame << std::endl;
        return 1;
    }
    
    decoder.close();
    std::cout << "Processing completed successfully." << std::endl;
    
    return 0;
}