#pragma once

#include "audio_decoder.h"
#include "mkv_stream_reader.h"
#include "ring_buffer.h"
#include "audio_format_converter.h"
#include <memory>
#include <string>
#include <mutex>
#include <atomic>
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

extern "C" {
#include <libavutil/frame.h>
}


/**
 * @class AudioPlayer
 * @brief 音频播放器，支持时间驱动的音频播放和预解码缓冲
 * 
 * 特性：
 * - 基于 AudioDecoder 提供音频解码
 * - 支持时间驱动的音频播放控制 (onTimer)
 * - 预解码缓冲机制，提前储备音频帧
 * - 顺序播放，不支持跳转
 * - 线程安全的音频缓冲管理
 * - 内置 WASAPI 音频播放支持
 */
class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();
    
    /**
     * @brief 打开音频文件并初始化解码器
     * @param filepath 音频文件路径
     * @return true 成功打开音频文件，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 初始化音频播放器
     * @return true 成功初始化音频播放器，false 初始化失败
     */
    bool initialize();
    
    /**
     * @brief 定时器回调，更新音频播放状态
     * @param current_time 当前时间 (秒)，从播放开始计算的相对时间
     */
    void onTimer(double current_time);
    
    /**
     * @brief 关闭播放器并释放资源
     */
    void close();
    
    /**
     * @brief 检查播放器是否已打开文件
     * @return true 播放器已打开文件，false 播放器未打开文件
     */
    bool isOpen() const;
    
    /**
     * @brief 检查播放器是否已准备好播放
     * @return true 播放器已准备好，false 播放器未准备好
     */
    bool isReady() const;
    
    /**
     * @brief 检查是否到达文件末尾
     * @return true 到达文件末尾，false 未到达文件末尾
     */
    bool isEOF() const;
    
    /**
     * @brief 获取内部的音频解码器实例
     * @return AudioDecoder* 音频解码器指针，可能为 nullptr
     */
    AudioDecoder* getAudioDecoder() const;
    
    /**
     * @brief 获取音频缓冲区状态（用于测试验证）
     * @param total_frames 输出：缓冲区总帧数
     * @param padding_frames 输出：缓冲区中待播放帧数
     * @param available_frames 输出：可写入帧数
     * @return true 成功获取状态，false 获取失败
     */
    bool getAudioBufferStatus(UINT32& total_frames, UINT32& padding_frames, UINT32& available_frames) const;
    

private:
    std::unique_ptr<AudioDecoder> audio_decoder_;
    
    // 音频缓冲管理
    std::unique_ptr<RingBuffer> ring_buffer_;
    std::unique_ptr<AudioFormatConverter> format_converter_;
    mutable std::mutex audio_mutex_;
    
    // 流信息
    MKVStreamReader::StreamInfo stream_info_;
    
    // WASAPI 音频播放相关成员
    ComPtr<IMMDevice> audio_device_;
    ComPtr<IAudioClient> audio_client_;
    ComPtr<IAudioRenderClient> render_client_;
    UINT32 buffer_frame_count_;
    UINT32 sample_rate_;
    UINT16 audio_channels_;
    UINT16 bits_per_sample_;
    std::atomic<bool> is_audio_initialized_;
    std::atomic<bool> is_audio_playing_;
    
    // 私有方法
    
    /**
     * @brief 处理音频帧
     */
    void handleAudioFrames();
    
    
    /**
     * @brief 填充环形缓冲区
     */
    void fillRingBuffer();
    
    /**
     * @brief 从环形缓冲区写入到WASAPI
     */
    void writeToWASAPI();
    
    
    /**
     * @brief 将音频 PTS 转换为秒数
     * @param frame AVFrame 指针
     * @return double 时间戳（秒）
     */
    double convertAudioPtsToSeconds(AVFrame* frame) const;
    
    
    /**
     * @brief 初始化 WASAPI 音频播放器
     * @param sample_rate 采样率
     * @param channels 声道数
     * @param bits_per_sample 位深度
     * @return true 成功初始化，false 初始化失败
     */
    bool initializeWASAPIAudioPlayer(UINT32 sample_rate, UINT16 channels, UINT16 bits_per_sample);
    
    
    /**
     * @brief 写入音频数据到缓冲区
     * @param frame 音频帧
     * @return true 成功写入，false 写入失败
     */
    bool writeAudioData(AVFrame* frame);
    
    
    /**
     * @brief 启动音频播放器
     * @return true 成功启动，false 启动失败
     */
    bool startAudioPlayer();
    
    
    /**
     * @brief 停止音频播放器
     */
    void stopAudioPlayer();
    
    
    /**
     * @brief 关闭音频播放器
     */
    void closeAudioPlayer();
    
    
    /**
     * @brief 初始化音频设备
     * @return true 成功初始化，false 初始化失败
     */
    bool initializeAudioDevice();
    
    
    /**
     * @brief 配置音频客户端
     * @return true 成功配置，false 配置失败
     */
    bool configureAudioClient();
    
    
    /**
     * @brief 清理音频资源
     */
    void cleanupAudioResources();
    
    
    /**
     * @brief 将浮点音频数据转换为 PCM 格式
     * @param frame 音频帧
     * @param buffer 输出缓冲区
     * @param frames_to_write 要写入的帧数
     * @return true 成功转换，false 转换失败
     */
    bool convertFloatToPcm(AVFrame* frame, BYTE* buffer, UINT32 frames_to_write);
    
    /**
     * @brief 填充静音以防止 underrun
     * @param frame_count 要填充的帧数
     */
    void fillSilence(UINT32 frame_count);
}; 