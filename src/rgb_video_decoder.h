#ifndef RGB_VIDEO_DECODER_H
#define RGB_VIDEO_DECODER_H

#include "hw_video_decoder.h"
#include <memory>
#include <string>
#include <unordered_map>

// D3D11 headers are already included through hw_video_decoder.h

using Microsoft::WRL::ComPtr;

/**
 * @class RgbVideoDecoder
 * @brief RGB 视频解码器，基于 HwVideoDecoder 并提供 RGB 转换功能
 * 
 * 特性：
 * - 内部使用 HwVideoDecoder 进行硬件解码
 * - 使用 VideoProcessorBlt 将 YUV 格式转换为 RGB
 * - 提供 D3D11 RGB 纹理输出
 * - 自动管理 D3D11 资源
 */
class RgbVideoDecoder {
public:
    struct DecodedRgbFrame {
        HwVideoDecoder::DecodedFrame hw_frame;    // 原始硬件解码帧
        ComPtr<ID3D11Texture2D> rgb_texture;      // RGB D3D11 纹理
        bool is_valid;
    };
    
    RgbVideoDecoder();
    ~RgbVideoDecoder();
    
    /**
     * @brief 打开视频文件并初始化解码器
     * @param filepath 视频文件路径
     * @return true 成功打开并初始化，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 读取下一个解码帧并转换为 RGB
     * @param frame 用于存储解码帧的结构体
     * @return true 成功读取帧，false 读取失败或到达文件末尾
     * @note 调用者负责在使用完毕后调用 av_frame_free(&frame.hw_frame.frame) 释放内存
     */
    bool readNextFrame(DecodedRgbFrame& frame);
    
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
     * @brief 获取内部的硬件解码器实例
     * @return HwVideoDecoder* 硬件解码器指针，可能为 nullptr
     */
    HwVideoDecoder* getHwDecoder() const;
    
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
    std::unique_ptr<HwVideoDecoder> hw_decoder_;
    
    // D3D11 Video Processor 相关
    ComPtr<ID3D11VideoDevice> video_device_;
    ComPtr<ID3D11VideoContext> video_context_;
    ComPtr<ID3D11VideoProcessor> video_processor_;
    ComPtr<ID3D11VideoProcessorEnumerator> video_processor_enumerator_;
    
    // RGB 纹理相关
    ComPtr<ID3D11Texture2D> rgb_texture_;
    ComPtr<ID3D11VideoProcessorOutputView> output_view_;
    
    // Input View 缓存相关
    struct InputViewKey {
        ID3D11Texture2D* texture;
        UINT array_slice;
        
        bool operator==(const InputViewKey& other) const {
            return texture == other.texture && array_slice == other.array_slice;
        }
    };
    
    struct InputViewKeyHash {
        std::size_t operator()(const InputViewKey& key) const {
            return std::hash<void*>()(key.texture) ^ (std::hash<UINT>()(key.array_slice) << 1);
        }
    };
    
    std::unordered_map<InputViewKey, ComPtr<ID3D11VideoProcessorInputView>, InputViewKeyHash> input_view_cache_;
    
    // 视频尺寸
    int video_width_;
    int video_height_;
    
    bool initializeVideoProcessor();
    bool convertYuvToRgb(AVFrame* yuv_frame, ComPtr<ID3D11Texture2D>& rgb_texture);
    ComPtr<ID3D11VideoProcessorInputView> getOrCreateInputView(ID3D11Texture2D* texture, UINT array_slice);
    void clearInputViewCache();
    void cleanup();
};

#endif // RGB_VIDEO_DECODER_H 