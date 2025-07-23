#pragma once

#include "stereo_video_decoder.h"
#include "mkv_stream_reader.h"
#include <memory>
#include <string>
#include <queue>
#include <mutex>
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
 * - 基于 StereoVideoDecoder 提供立体视觉纹理输出
 * - 支持时间驱动的播放控制 (onTimer)
 * - 预解码缓冲机制，提前储备下一帧
 * - 顺序播放，不支持跳转
 * - 线程安全的帧缓冲管理
 * - 支持测试和生产环境两种渲染目标
 * - 纯视频播放，不处理音频
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
     * @brief 获取内部的立体解码器实例
     * @return StereoVideoDecoder* 立体解码器指针，可能为 nullptr
     */
    StereoVideoDecoder* getStereoDecoder() const;
    
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
    std::unique_ptr<StereoVideoDecoder> stereo_decoder_;
    
    // 帧缓冲管理
    std::queue<DecodedStereoFrame> frame_buffer_;
    ComPtr<ID3D11Texture2D> current_frame_texture_;
    double current_frame_time_ = 0.0;
    
    // StereoVideoDecoder 不暴露流信息
    
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
    
    // 私有方法
    
    
    /**
     * @brief 预加载下一帧到缓冲区
     * @return true 成功预加载，false 预加载失败
     */
    bool preloadNextFrame();
    
    
    // convertPtsToSeconds 函数已移除，直接从 AVFrame 和时间基计算时间戳
    
    
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
};
