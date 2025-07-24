#pragma once

#include <memory>
#include <string>
#include <d3d11.h>
#include <wrl/client.h>
#include "float_rgb_video_decoder.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/rational.h>
}

// TensorRT 前向声明 - 使用结构体包装参数
struct TensorDims {
    int nbDims;
    int d[8];  // TensorRT 最大维度数
};

struct DecodedStereoFrame {
    FloatRgbVideoDecoder::DecodedFloatRgbFrame* input_frame = nullptr;  // 输入的RGB帧
    bool is_valid = false;
    
    // TensorRT 立体视觉输出
    // 像素格式：32位浮点数 (float32)
    // 颜色布局：单通道深度图 (1 channel depth map) 
    // 数据范围：[0.0, 1.0] 归一化深度值，0.0=最近，1.0=最远
    // 内存布局：行优先 (row-major)，连续存储
    // 获取尺寸：通过 getWidth()/getHeight() 获取
    // 缓冲区大小：getWidth() * getHeight() * sizeof(float)
    void* cuda_output_buffer = nullptr;  // CUDA 设备内存指针
};

class StereoVideoDecoder {
public:
    StereoVideoDecoder();
    ~StereoVideoDecoder();
    
    bool open(const std::string& filepath);
    void close();
    bool isOpen() const;
    bool isEOF() const;
    
    bool readNextFrame(DecodedStereoFrame& frame);
    
    int getWidth() const;
    int getHeight() const;
    ID3D11Device* getD3D11Device() const;
    AVRational getVideoTimeBase() const;
    
private:
    std::unique_ptr<FloatRgbVideoDecoder> float_rgb_decoder_;
    FloatRgbVideoDecoder::DecodedFloatRgbFrame current_input_frame_;  // 存储当前输入帧
    bool is_open_;
    
    // TensorRT 成员
    void* tensorrt_runtime_ = nullptr;
    void* tensorrt_engine_ = nullptr;
    void* tensorrt_context_ = nullptr;
    void* tensorrt_logger_ = nullptr;
    
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
    size_t device_input_size_ = 0;
    size_t device_output_size_ = 0;
    
    
    bool tensorrt_initialized_ = false;
    
    bool initializeTensorRT();
    bool convertToStereo(void* input_cuda_ptr);
    void cleanupTensorRT();
    
    // 拆分后的辅助函数
    bool prepareInferenceInput(void* input_cuda_ptr, const TensorDims& runtime_dims);
    bool executeInference();
};