#pragma once

#include "rgb_video_decoder.h"
#include <memory>
#include <string>

// D3D11 headers are already included through rgb_video_decoder.h
#include <d3dcompiler.h>

using Microsoft::WRL::ComPtr;

/**
 * @class FloatRgbVideoDecoder
 * @brief 浮点RGB视频解码器，基于 RgbVideoDecoder 并提供CUDA兼容的浮点格式输出
 * 
 * 特性：
 * - 内部使用 RgbVideoDecoder 进行 RGB 解码
 * - 将 DXGI_FORMAT_B8G8R8A8_UNORM 转换为浮点RGBA格式(BCHW布局)
 * - 输出CUDA兼容的线性缓冲区，可直接用于TensorRT推理
 * - 自动管理 D3D11 和 CUDA 互操作资源
 */
class FloatRgbVideoDecoder {
public:
    struct DecodedFloatRgbFrame {
        RgbVideoDecoder::DecodedRgbFrame rgb_frame;    // 原始RGB解码帧
        void* cuda_buffer;                             // CUDA设备指针，BCHW float32 RGBA格式
        size_t buffer_size;                            // 缓冲区大小(字节)
        bool is_valid;
    };
    
    FloatRgbVideoDecoder();
    ~FloatRgbVideoDecoder();
    
    /**
     * @brief 打开视频文件并初始化解码器
     * @param filepath 视频文件路径
     * @return true 成功打开并初始化，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 读取下一个解码帧并转换为CUDA兼容的浮点RGBA缓冲区
     * @param frame 用于存储解码帧的结构体，包含CUDA设备指针
     * @return true 成功读取帧，false 读取失败或到达文件末尾
     * @note 调用者负责在使用完毕后调用 av_frame_free(&frame.rgb_frame.hw_frame.frame) 释放内存
     */
    bool readNextFrame(DecodedFloatRgbFrame& frame);
    
    /**
     * @brief 检查解码器是否已打开
     * @return true 解码器已打开，false 解码器未打开
     */
    bool isOpen() const;
    
    /**
     * @brief 检查是否到达文件末尾
     * @return true 到达文件末尾，false 未到达文件末尾
     */
    bool isEOF() const;
    
    /**
     * @brief 关闭解码器并释放资源
     */
    void close();
    
    /**
     * @brief 获取内部的RGB解码器实例
     * @return RgbVideoDecoder* RGB解码器指针，可能为 nullptr
     */
    RgbVideoDecoder* getRgbDecoder() const;
    
    /**
     * @brief 获取视频宽度
     * @return int 视频宽度
     */
    int getWidth() const;
    
    /**
     * @brief 获取视频高度
     * @return int 视频高度
     */
    int getHeight() const;
    
    /**
     * @brief 获取当前帧的CUDA设备指针
     * @return void* CUDA设备指针，格式为BCHW float32 RGBA
     */
    void* getCudaBuffer() const;
    
    /**
     * @brief 获取CUDA缓冲区大小
     * @return size_t 缓冲区大小(字节)
     */
    size_t getBufferSize() const;
    
private:
    std::unique_ptr<RgbVideoDecoder> rgb_decoder_;
    
    // D3D11 设备相关
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ShaderResourceView> input_srv_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    ComPtr<ID3D11ComputeShader> conversion_shader_;
    
    // CUDA互操作相关
    ComPtr<ID3D11Buffer> cuda_buffer_;         // D3D11 buffer，用作CUDA互操作
    ComPtr<ID3D11Buffer> constant_buffer_;     // 常量缓冲区，用于传递width/height
    void* cuda_graphics_resource_;             // CUDA图形资源句柄
    void* cuda_device_ptr_;                    // 映射的CUDA设备指针
    size_t buffer_size_;                       // 缓冲区大小
    
    // 视频尺寸
    int video_width_;
    int video_height_;
    
    bool initializeConversionResources();
    bool convertToCudaBuffer(ComPtr<ID3D11Texture2D> rgb_texture, void*& cuda_ptr);
    void cleanup();
};