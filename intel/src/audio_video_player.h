#pragma once
#include "audio_player.h"
#include "video_player.h"
#include <memory>
#include <string>
#include <wrl/client.h>
#include <dxgi.h>
using Microsoft::WRL::ComPtr;
class AudioVideoPlayer {public: AudioVideoPlayer(); ~AudioVideoPlayer(); bool open(const std::string& filepath); bool setRenderTarget(ComPtr<ID3D11Texture2D> rt); bool setRenderTarget(ComPtr<IDXGISwapChain> sc); bool initializeAudio(); void onTimer(double current_time); ComPtr<ID3D11Texture2D> getCurrentFrame() const; void close(); bool isOpen() const; bool isVideoReady() const; bool isAudioReady() const; bool hasAudio() const; bool isEOF() const; AudioDecoder* getAudioDecoder() const; VideoPlayer* getVideoPlayer() const; AudioPlayer* getAudioPlayer() const; ID3D11Device* getD3D11Device() const; ID3D11DeviceContext* getD3D11DeviceContext() const; bool getAudioBufferStatus(UINT32& total, UINT32& padding, UINT32& available) const; private: std::unique_ptr<VideoPlayer> video_player_; std::unique_ptr<AudioPlayer> audio_player_; bool has_audio_ = false; std::string filepath_; };
