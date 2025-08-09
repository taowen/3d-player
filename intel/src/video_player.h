#pragma once
#include "rgb_video_decoder.h"
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <wrl/client.h>
#include <dxgi.h>
using Microsoft::WRL::ComPtr;

enum class RenderTargetType { TEXTURE, SWAPCHAIN };

class VideoPlayer {
public:
    VideoPlayer();
    ~VideoPlayer();
    bool open(const std::string& filepath);
    bool setRenderTarget(ComPtr<ID3D11Texture2D> render_target);
    bool setRenderTarget(ComPtr<IDXGISwapChain> swap_chain);
    void onTimer(double current_time);
    ComPtr<ID3D11Texture2D> getCurrentFrame() const;
    void close();
    bool isOpen() const; bool isReady() const; bool isEOF() const;
    RgbVideoDecoder* getRgbDecoder() const; ID3D11Device* getD3D11Device() const; ID3D11DeviceContext* getD3D11DeviceContext() const;
    int getWidth() const; int getHeight() const;
private:
    std::unique_ptr<RgbVideoDecoder> rgb_decoder_;
    struct DecodedRgbFrameWrapper { RgbVideoDecoder::DecodedRgbFrame frame; ComPtr<ID3D11Texture2D> d3d_texture; };
    std::queue<DecodedRgbFrameWrapper> frame_buffer_;
    ComPtr<ID3D11Texture2D> current_frame_texture_; double current_frame_time_ = 0.0;
    RenderTargetType render_target_type_{}; ComPtr<ID3D11Texture2D> render_target_texture_; ComPtr<IDXGISwapChain> render_target_swapchain_; ComPtr<ID3D11RenderTargetView> render_target_view_;
    mutable std::mutex frame_mutex_;
    ComPtr<ID3D11VertexShader> fullscreen_vs_; ComPtr<ID3D11PixelShader> fullscreen_ps_; ComPtr<ID3D11SamplerState> texture_sampler_; ComPtr<ID3D11BlendState> blend_state_; ComPtr<ID3D11RasterizerState> raster_state_;
    bool preloadNextFrame(); bool initializeRenderTarget(ComPtr<ID3D11Texture2D> rt); bool initializeRenderTarget(ComPtr<IDXGISwapChain> sc); bool initializeRenderPipeline(ID3D11Device* device); void renderFrame(ComPtr<ID3D11Texture2D> source_texture);
};
