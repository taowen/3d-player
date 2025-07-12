#pragma once

#include "rgb_video_decoder.h"
#include "mkv_stream_reader.h"
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <atomic>
#include <dxgi.h>

// D3D11 headers are already included through rgb_video_decoder.h

using Microsoft::WRL::ComPtr;


/**
 * @brief 渲染目标类型
 */
enum class RenderTargetType {
    TEXTURE,      // 渲染到纹理（测试环境）
    SWAPCHAIN     // 渲染到交换链（生产环境）
};


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
 * - 支持测试和生产环境两种渲染目标
 */
class VideoPlayer {
public:
    VideoPlayer();
    ~VideoPlayer();
    
    /**
     * @brief 打开视频文件并初始化播放器（测试环境：渲染到纹理）
     * @param filepath 视频文件路径
     * @param render_target 渲染目标纹理
     * @return true 成功打开视频文件，false 打开失败
     */
    bool open(const std::string& filepath, ComPtr<ID3D11Texture2D> render_target);
    
    /**
     * @brief 打开视频文件并初始化播放器（生产环境：渲染到交换链）
     * @param filepath 视频文件路径
     * @param swap_chain 交换链
     * @return true 成功打开视频文件，false 打开失败
     */
    bool open(const std::string& filepath, ComPtr<IDXGISwapChain> swap_chain);
    
    /**
     * @brief 定时器回调，更新播放状态并内部进行渲染
     * @param current_time 当前时间 (秒)，从播放开始计算的相对时间
     * @note 调用频率应该高于视频帧率，由外部控制
     * @note 内部会自动进行渲染，无需外部调用渲染方法
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
    std::unique_ptr<RgbVideoDecoder> rgb_decoder_;
    
    // 帧缓冲管理
    std::queue<RgbVideoDecoder::DecodedRgbFrame> frame_buffer_;
    ComPtr<ID3D11Texture2D> current_frame_texture_;
    double current_frame_time_ = 0.0;
    
    // 视频流信息
    MKVStreamReader::StreamInfo stream_info_;
    
    // 渲染目标管理
    RenderTargetType render_target_type_;
    ComPtr<ID3D11Texture2D> render_target_texture_;     // 测试环境：目标纹理
    ComPtr<IDXGISwapChain> render_target_swapchain_;    // 生产环境：交换链
    ComPtr<ID3D11RenderTargetView> render_target_view_; // 渲染目标视图
    
    // 线程安全
    mutable std::mutex frame_mutex_;
    
    // 私有方法
    
    
    /**
     * @brief 预加载下一帧到缓冲区
     * @return true 成功预加载，false 预加载失败
     */
    bool preloadNextFrame();
    
    
    /**
     * @brief 将 AVFrame 的 PTS 转换为秒数
     * @param frame AVFrame 指针
     * @return double 时间戳（秒）
     */
    double convertPtsToSeconds(AVFrame* frame) const;
    
    
    /**
     * @brief 初始化渲染目标（纹理）
     * @param render_target 渲染目标纹理
     * @return true 成功初始化，false 初始化失败
     */
    bool initializeRenderTarget(ComPtr<ID3D11Texture2D> render_target);
    
    
    /**
     * @brief 初始化渲染目标（交换链）
     * @param swap_chain 交换链
     * @return true 成功初始化，false 初始化失败
     */
    bool initializeRenderTarget(ComPtr<IDXGISwapChain> swap_chain);
    
    
    /**
     * @brief 执行渲染操作
     * @param source_texture 源纹理
     */
    void renderFrame(ComPtr<ID3D11Texture2D> source_texture);
};
