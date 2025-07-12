#pragma once

#include "rgb_video_decoder.h"
#include "audio_decoder.h"
#include "mkv_stream_reader.h"
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <atomic>
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <wrl/client.h>
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
 * - 内置 WASAPI 音频播放支持
 */
class VideoPlayer {
public:
    VideoPlayer();
    ~VideoPlayer();
    
    /**
     * @brief 打开视频文件并初始化解码器（不设置渲染目标）
     * @param filepath 视频文件路径
     * @return true 成功打开视频文件，false 打开失败
     * @note 打开后可以通过 getD3D11Device() 获取设备，然后调用 setRenderTarget() 设置渲染目标
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
     * @brief 检查播放器是否已打开文件
     * @return true 播放器已打开文件，false 播放器未打开文件
     */
    bool isOpen() const;
    
    /**
     * @brief 检查播放器是否已准备好播放（文件已打开且渲染目标已设置）
     * @return true 播放器已准备好，false 播放器未准备好
     */
    bool isReady() const;
    
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
    
    /**
     * @brief 获取 D3D11 设备
     * @return ID3D11Device* D3D11 设备指针，可能为 nullptr
     * @note 只有在 open() 成功后才能获取到设备
     */
    ID3D11Device* getD3D11Device() const;
    
    /**
     * @brief 获取 D3D11 设备上下文
     * @return ID3D11DeviceContext* D3D11 设备上下文指针，可能为 nullptr
     * @note 只有在 open() 成功后才能获取到设备上下文
     */
    ID3D11DeviceContext* getD3D11DeviceContext() const;

    /**
     * @brief 检查是否有音频流
     * @return true 有音频流，false 没有音频流
     */
    bool hasAudio() const;
    
    /**
     * @brief 初始化音频播放器
     * @return true 成功初始化音频播放器，false 初始化失败
     */
    bool initializeAudio();
    
    /**
     * @brief 检查音频播放器是否准备就绪
     * @return true 音频播放器准备就绪，false 音频播放器未准备就绪
     */
    bool isAudioReady() const;
    
    /**
     * @brief 获取音频缓冲区状态（用于测试验证）
     * @param total_frames 输出：缓冲区总帧数
     * @param padding_frames 输出：缓冲区中待播放帧数
     * @param available_frames 输出：可写入帧数
     * @return true 成功获取状态，false 获取失败
     */
    bool getAudioBufferStatus(UINT32& total_frames, UINT32& padding_frames, UINT32& available_frames) const;

private:
    std::unique_ptr<RgbVideoDecoder> rgb_decoder_;
    std::unique_ptr<AudioDecoder> audio_decoder_;        // 音频解码器
    
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
    
    // 渲染管线资源
    ComPtr<ID3D11VertexShader> fullscreen_vs_;
    ComPtr<ID3D11PixelShader> fullscreen_ps_;
    ComPtr<ID3D11SamplerState> texture_sampler_;
    ComPtr<ID3D11BlendState> blend_state_;
    ComPtr<ID3D11RasterizerState> raster_state_;
    ComPtr<ID3D11ShaderResourceView> current_texture_srv_;
    
    // 音频相关成员
    std::queue<AudioDecoder::DecodedFrame> audio_buffer_;
    mutable std::mutex audio_mutex_;
    bool has_audio_;
    
    // WASAPI 音频播放相关成员（合并自 WASAPIAudioPlayer）
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
     * @brief 初始化渲染管线资源
     * @param device D3D11 设备
     * @return true 成功初始化，false 初始化失败
     */
    bool initializeRenderPipeline(ID3D11Device* device);


    /**
     * @brief 为纹理创建 Shader Resource View
     * @param texture 源纹理
     * @return ComPtr<ID3D11ShaderResourceView> 创建的 SRV
     */
    ComPtr<ID3D11ShaderResourceView> createTextureShaderResourceView(ComPtr<ID3D11Texture2D> texture);


    /**
     * @brief 执行渲染操作
     * @param source_texture 源纹理
     */
    void renderFrame(ComPtr<ID3D11Texture2D> source_texture);

    /**
     * @brief 处理音频帧（在 onTimer 中调用）
     * @param current_time 当前时间
     */
    void handleAudioFrames(double current_time);
    
    /**
     * @brief 预加载音频帧到缓冲区
     */
    void preloadAudioFrames();
    
    /**
     * @brief 将音频帧PTS转换为秒数
     * @param frame 音频帧
     * @return 时间（秒）
     */
    double convertAudioPtsToSeconds(AVFrame* frame) const;
    
    // WASAPI 音频播放相关私有方法（合并自 WASAPIAudioPlayer）
    
    
    /**
     * @brief 初始化 WASAPI 音频播放器
     * @param sample_rate 采样率 (如 44100)
     * @param channels 声道数 (如 2)
     * @param bits_per_sample 位深度 (如 16)
     * @return true 成功初始化，false 初始化失败
     */
    bool initializeWASAPIAudioPlayer(UINT32 sample_rate, UINT16 channels, UINT16 bits_per_sample);
    
    
    /**
     * @brief 非阻塞写入音频数据
     * @param frame AVFrame 音频帧数据
     * @return true 成功写入，false 缓冲区满或写入失败
     * @note 此方法不会阻塞，如果缓冲区满则立即返回false
     */
    bool writeAudioData(AVFrame* frame);
    
    
    /**
     * @brief 开始音频播放
     * @return true 成功开始播放，false 开始失败
     */
    bool startAudioPlayer();
    
    
    /**
     * @brief 停止音频播放
     */
    void stopAudioPlayer();
    
    
    /**
     * @brief 关闭音频播放器并释放资源
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
     * @brief 音频格式转换辅助函数
     * @param frame 源音频帧
     * @param buffer 目标缓冲区
     * @param frames_to_write 要写入的帧数
     * @return true 成功转换，false 转换失败
     */
    bool convertFloatToPcm(AVFrame* frame, BYTE* buffer, UINT32 frames_to_write);
};
