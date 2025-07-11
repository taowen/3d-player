#ifndef HW_VIDEO_DECODER_H
#define HW_VIDEO_DECODER_H

#include "mkv_stream_reader.h"
#include <memory>
#include <string>

// Include D3D11 headers before FFmpeg headers
#define WIN32_LEAN_AND_MEAN
#include <d3d11.h>
#include <wrl/client.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}

/**
 * @class HwVideoDecoder
 * @brief DirectX 11 硬件视频解码器，使用 FFmpeg AVBufferPool 进行内存管理
 * 
 * 特性：
 * - 必须使用硬件解码，无软件解码回退
 * - 支持 H.264 (h264_d3d11va)、HEVC (hevc_d3d11va)、AV1 (av1_d3d11va)
 * - 输出格式为 AV_PIX_FMT_YUV444P (DirectX 11纹理)
 * - 使用 FFmpeg AVBufferPool 自动管理D3D11纹理缓冲区
 */
class HwVideoDecoder {
public:
    struct DecodedFrame {
        AVFrame* frame;        // D3D11 硬件帧 (AV_PIX_FMT_YUV444P)
        bool is_valid;
    };
    
    HwVideoDecoder();
    ~HwVideoDecoder();
    
    /**
     * @brief 打开视频文件并初始化硬件解码器
     * @param filepath 视频文件路径
     * @return true 成功打开并初始化硬件解码器，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 读取下一个解码帧 (使用 FFmpeg AVBufferPool 自动管理)
     * @param frame 用于存储解码帧的结构体
     * @return true 成功读取帧，false 读取失败或到达文件末尾
     * @note 调用者负责在使用完毕后调用 av_frame_free(&frame.frame) 释放内存
     */
    bool readNextFrame(DecodedFrame& frame);
    
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
     * @brief 获取内部的流读取器实例
     * @return MKVStreamReader* 流读取器指针，可能为 nullptr
     */
    MKVStreamReader* getStreamReader() const;
    
    /**
     * @brief 获取 D3D11 设备接口
     * @return ID3D11Device* D3D11设备指针，可能为 nullptr
     */
    ID3D11Device* getD3D11Device() const;
    
    /**
     * @brief 获取 D3D11 设备上下文接口
     * @return ID3D11DeviceContext* D3D11设备上下文指针，可能为 nullptr
     */
    ID3D11DeviceContext* getD3D11DeviceContext() const;
    
private:
    std::unique_ptr<MKVStreamReader> stream_reader_;
    AVCodecContext* codec_context_;
    AVBufferRef* hw_device_ctx_;   // D3D11VA 硬件设备上下文
    
    bool initializeHardwareDecoder();
    bool processPacket(AVPacket* packet, DecodedFrame& frame);
    void cleanup();
};

#endif // HW_VIDEO_DECODER_H