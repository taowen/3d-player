#pragma once

#include "audio_player.h"
#include "video_player.h"
#include <memory>
#include <string>
#include <wrl/client.h>
#include <dxgi.h>

using Microsoft::WRL::ComPtr;


/**
 * @class AudioVideoPlayer
 * @brief 音视频播放器，组合音频播放器和视频播放器功能
 * 
 * 特性：
 * - 组合 AudioPlayer 和 VideoPlayer
 * - 支持时间驱动的音视频同步播放
 * - 统一的接口管理音频和视频播放
 * - 顺序播放，不支持跳转
 * - 支持测试和生产环境两种渲染目标
 */
class AudioVideoPlayer {
public:
    AudioVideoPlayer();
    ~AudioVideoPlayer();
    
    /**
     * @brief 打开音视频文件并初始化解码器
     * @param filepath 音视频文件路径
     * @return true 成功打开文件，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 设置渲染目标（测试环境：渲染到纹理）
     * @param render_target 渲染目标纹理
     * @return true 成功设置渲染目标，false 设置失败
     * @note 必须在 open() 之后调用
     */
    bool setRenderTarget(ComPtr<ID3D11Texture2D> render_target);
    
    /**
     * @brief 设置渲染目标（生产环境：渲染到交换链）
     * @param swap_chain 交换链
     * @return true 成功设置渲染目标，false 设置失败
     * @note 必须在 open() 之后调用
     */
    bool setRenderTarget(ComPtr<IDXGISwapChain> swap_chain);
    
    /**
     * @brief 初始化音频播放器
     * @return true 成功初始化音频播放器，false 初始化失败
     */
    bool initializeAudio();
    
    /**
     * @brief 定时器回调，更新音视频播放状态
     * @param current_time 当前时间 (秒)，从播放开始计算的相对时间
     * @note 调用频率应该高于视频帧率，由外部控制
     */
    void onTimer(double current_time);
    
    /**
     * @brief 获取当前帧的 RGB 纹理（仅用于测试验证）
     * @return ComPtr<ID3D11Texture2D> 当前帧的 RGB 纹理，可能为空
     */
    ComPtr<ID3D11Texture2D> getCurrentFrame() const;
    
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
     * @brief 检查视频播放器是否已准备好播放
     * @return true 视频播放器已准备好，false 视频播放器未准备好
     */
    bool isVideoReady() const;
    
    /**
     * @brief 检查音频播放器是否已准备好播放
     * @return true 音频播放器已准备好，false 音频播放器未准备好
     */
    bool isAudioReady() const;
    
    /**
     * @brief 检查是否有音频流
     * @return true 有音频流，false 没有音频流
     */
    bool hasAudio() const;
    
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
     * @brief 获取视频播放器实例
     * @return VideoPlayer* 视频播放器指针，可能为 nullptr
     */
    VideoPlayer* getVideoPlayer() const;
    
    /**
     * @brief 获取音频播放器实例
     * @return AudioPlayer* 音频播放器指针，可能为 nullptr
     */
    AudioPlayer* getAudioPlayer() const;
    
    /**
     * @brief 获取 D3D11 设备
     * @return ID3D11Device* D3D11 设备指针，可能为 nullptr
     */
    ID3D11Device* getD3D11Device() const;
    
    /**
     * @brief 获取 D3D11 设备上下文
     * @return ID3D11DeviceContext* D3D11 设备上下文指针，可能为 nullptr
     */
    ID3D11DeviceContext* getD3D11DeviceContext() const;
    
    /**
     * @brief 获取音频缓冲区状态（用于测试验证）
     * @param total_frames 输出：缓冲区总帧数
     * @param padding_frames 输出：缓冲区中待播放帧数
     * @param available_frames 输出：可写入帧数
     * @return true 成功获取状态，false 获取失败
     */
    bool getAudioBufferStatus(UINT32& total_frames, UINT32& padding_frames, UINT32& available_frames) const;

private:
    std::unique_ptr<VideoPlayer> video_player_;
    std::unique_ptr<AudioPlayer> audio_player_;
    bool has_audio_;
    std::string filepath_;
}; 