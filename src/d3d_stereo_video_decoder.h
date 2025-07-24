#pragma once

#include <memory>
#include <string>
#include <d3d11.h>
#include <wrl/client.h>
#include "stereo_video_decoder.h"

extern "C" {
#include <libavutil/rational.h>
}

// CUDA headers
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

using Microsoft::WRL::ComPtr;

struct DecodedStereoFrameD3D {
    DecodedStereoFrame stereo_frame;           // 来自 StereoVideoDecoder 的原始输出
    
    // D3D11 立体视觉纹理输出
    // 纹理格式：DXGI_FORMAT_R32G32B32A32_FLOAT (32位浮点 RGBA)
    // 纹理布局：Half Side-by-Side 立体图像
    // ├─ 左半部分：左眼视图 (0, 0) 到 (width/2, height)
    // └─ 右半部分：右眼视图 (width/2, 0) 到 (width, height)
    // 数据范围：[0.0, 1.0] 归一化RGB像素值，A通道固定为1.0
    // 纹理尺寸：与原始视频相同 (getWidth() x getHeight())
    // 内存布局：标准 D3D11 纹理格式，支持 Shader Resource View
    ComPtr<ID3D11Texture2D> d3d_texture;
    
    bool d3d_conversion_valid = false;         // D3D11 转换是否成功
};

class D3dStereoVideoDecoder {
public:
    D3dStereoVideoDecoder();
    ~D3dStereoVideoDecoder();
    
    /**
     * @brief 打开视频文件并初始化解码器
     * @param filepath 视频文件路径
     * @return true 成功打开并初始化，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 关闭解码器并释放资源
     */
    void close();
    
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
    
    bool readNextFrame(DecodedStereoFrameD3D& frame);
    
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
     * @brief 获取 D3D11 设备
     * @return ID3D11Device* D3D11 设备指针
     */
    ID3D11Device* getD3D11Device() const;
    
    /**
     * @brief 获取视频时间基准
     * @return AVRational 视频时间基准
     */
    AVRational getVideoTimeBase() const;

private:
    std::unique_ptr<StereoVideoDecoder> stereo_decoder_;
    
    // CUDA-D3D11 互操作资源
    ComPtr<ID3D11Texture2D> d3d_texture_;
    cudaGraphicsResource* cuda_graphics_resource_ = nullptr;
    bool interop_initialized_ = false;
    
    // 视频尺寸缓存
    int width_ = 0;
    int height_ = 0;
    
    /**
     * @brief 初始化 CUDA-D3D11 互操作
     * @return true 初始化成功，false 初始化失败
     */
    bool initializeCudaD3DInterop();
    
    /**
     * @brief 将 CUDA 输出转换为 D3D11 纹理
     * @param stereo_frame 包含 CUDA 输出的立体帧
     * @return true 转换成功，false 转换失败
     */
    bool convertCudaToD3DTexture(const DecodedStereoFrame& stereo_frame);
    
    /**
     * @brief 清理互操作资源
     */
    void cleanupInterop();
};