#pragma once

#include "rgb_video_decoder.h"
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <atomic>

// D3D11 headers are already included through rgb_video_decoder.h

using Microsoft::WRL::ComPtr;


/**
 * @class VideoPlayer
 * @brief 视频播放器，支持时间驱动的播放控制和预解码缓冲
 * 
 * 特性：
 * - 基于 RgbVideoDecoder 提供 RGB 纹理输出
 * - 支持时间驱动的播放控制 (onTimer)
 * - 预解码缓冲机制，提前储备下一帧
 * - 顺序播放，不支持跳转
 * - 线程安全的帧缓冲管理
 */
class VideoPlayer {
public:
    VideoPlayer();
    ~VideoPlayer();
    
    /**
     * @brief 打开视频文件并初始化播放器
     * @param filepath 视频文件路径
     * @return true 成功打开视频文件，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 定时器回调，更新播放状态并切换帧
     * @param current_time 当前时间 (秒)，从播放开始计算的相对时间
     * @return true 需要重新渲染 (切换到了新帧)，false 不需要渲染
     * @note 调用频率应该高于视频帧率，由外部控制
     */
    bool onTimer(double current_time);
    
    /**
     * @brief 获取当前帧的 RGB 纹理
     * @return ComPtr<ID3D11Texture2D> 当前帧的 RGB 纹理，可能为空
     */
    ComPtr<ID3D11Texture2D> getCurrentFrame() const;
    
    /**
     * @brief 关闭播放器并释放资源
     */
    void close();
    
    /**
     * @brief 检查播放器是否已打开
     * @return true 播放器已打开，false 播放器未打开
     */
    bool isOpen() const;
    
    /**
     * @brief 检查是否到达文件末尾
     * @return true 到达文件末尾，false 未到达文件末尾
     */
    bool isEOF() const;
    
    /**
     * @brief 获取内部的 RGB 解码器实例
     * @return RgbVideoDecoder* RGB 解码器指针，可能为 nullptr
     */
    RgbVideoDecoder* getRgbDecoder() const;

private:
    std::unique_ptr<RgbVideoDecoder> rgb_decoder_
};
