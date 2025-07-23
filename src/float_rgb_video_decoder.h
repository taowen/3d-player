#pragma once

#include "rgb_video_decoder.h"
#include <memory>
#include <string>

// D3D11 headers are already included through rgb_video_decoder.h
#include <d3dcompiler.h>

using Microsoft::WRL::ComPtr;

/**
 * @class FloatRgbVideoDecoder
 * @brief 浮点RGB视频解码器，基于 RgbVideoDecoder 并提供浮点格式转换功能
 * 
 * 特性：
 * - 内部使用 RgbVideoDecoder 进行 RGB 解码
 * - 将 DXGI_FORMAT_B8G8R8A8_UNORM 转换为 DXGI_FORMAT_R32G32B32A32_FLOAT
 * - 提供浮点精度的 D3D11 RGB 纹理输出
 * - 自动管理 D3D11 资源
 */
class FloatRgbVideoDecoder {
public:
    struct DecodedFloatRgbFrame {
        RgbVideoDecoder::DecodedRgbFrame rgb_frame;    // 原始RGB解码帧
        ComPtr<ID3D11Texture2D> float_texture;         // 浮点RGB D3D11纹理
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
     * @brief 读取下一个解码帧并转换为浮点RGB
     * @param frame 用于存储解码帧的结构体
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
    
private:
    std::unique_ptr<RgbVideoDecoder> rgb_decoder_;
    
    // D3D11 设备相关
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ShaderResourceView> input_srv_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    ComPtr<ID3D11ComputeShader> conversion_shader_;
    
    // 浮点纹理相关
    ComPtr<ID3D11Texture2D> float_texture_;
    
    // 视频尺寸
    int video_width_;
    int video_height_;
    
    bool initializeConversionResources();
    bool convertToFloat(ComPtr<ID3D11Texture2D> rgb_texture, ComPtr<ID3D11Texture2D>& float_texture);
    void cleanup();
};