#pragma once

#include "mkv_stream_reader.h"
#include <memory>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
}


/**
 * @class AudioDecoder
 * @brief 音频解码器，使用 FFmpeg 进行音频解码
 * 
 * 特性：
 * - 支持常见的音频编解码器（AAC、MP3、Opus等）
 * - 输出格式为 PCM 音频帧
 * - 使用 FFmpeg 自动管理音频缓冲区
 */
class AudioDecoder {
public:
    struct DecodedFrame {
        AVFrame* frame;        // 音频帧数据
        bool is_valid;
    };
    
    AudioDecoder();
    ~AudioDecoder();
    
    /**
     * @brief 打开视频文件并初始化音频解码器
     * @param filepath 视频文件路径
     * @return true 成功打开并初始化音频解码器，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 读取下一个解码帧
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
    
private:
    std::unique_ptr<MKVStreamReader> stream_reader_;
    AVCodecContext* codec_context_;
    
    bool initializeAudioDecoder();
    bool processPacket(AVPacket* packet, DecodedFrame& frame);
    void cleanup();
}; 