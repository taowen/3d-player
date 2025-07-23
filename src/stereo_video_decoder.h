#pragma once

#include <memory>
#include <string>
#include <d3d11.h>
#include <wrl/client.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/rational.h>
}

class FloatRgbVideoDecoder;

// TensorRT 前向声明 - 使用结构体包装参数
struct TensorDims {
    int nbDims;
    int d[8];  // TensorRT 最大维度数
};

struct DecodedStereoFrame {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> stereo_texture;
    AVFrame* frame = nullptr;  // 暴露完整frame，包含pts等所有信息
    bool is_valid = false;
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
    Microsoft::WRL::ComPtr<ID3D11Texture2D> stereo_texture_;
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
    
    void* input_resource_ = nullptr;
    void* output_resource_ = nullptr;
    ID3D11Texture2D* registered_input_texture_ = nullptr;
    ID3D11Texture2D* registered_output_texture_ = nullptr;
    
    bool tensorrt_initialized_ = false;
    
    bool initializeTensorRT();
    bool convertToStereo(ID3D11Texture2D* input_rgb, ID3D11Texture2D* output_stereo);
    void cleanupTensorRT();
    
    // 拆分后的辅助函数
    bool prepareInferenceInput(ID3D11Texture2D* input_rgb, const TensorDims& runtime_dims);
    bool executeInference();
    bool copyInferenceOutput(ID3D11Texture2D* output_stereo, const D3D11_TEXTURE2D_DESC& input_desc);
    bool registerCudaResources(ID3D11Texture2D* input_rgb, ID3D11Texture2D* output_stereo);
};