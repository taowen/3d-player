#include "audio_video_player.h"
#include <iostream>


AudioVideoPlayer::AudioVideoPlayer() 
    : video_player_(std::make_unique<VideoPlayer>())
    , audio_player_(std::make_unique<AudioPlayer>())
    , has_audio_(false) {
}


AudioVideoPlayer::~AudioVideoPlayer() {
    close();
}


bool AudioVideoPlayer::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    filepath_ = filepath;
    
    // 打开视频播放器
    if (!video_player_->open(filepath)) {
        std::cerr << "Failed to open video player for: " << filepath << std::endl;
        return false;
    }
    
    // 尝试打开音频播放器
    has_audio_ = audio_player_->open(filepath);
    if (!has_audio_) {
        std::cerr << "Warning: No audio stream found or failed to open audio player, will play video only" << std::endl;
    }
    
    return true;
}


bool AudioVideoPlayer::setRenderTarget(ComPtr<ID3D11Texture2D> render_target) {
    if (!video_player_) {
        std::cerr << "Cannot set render target: video player not initialized" << std::endl;
        return false;
    }
    
    return video_player_->setRenderTarget(render_target);
}


bool AudioVideoPlayer::setRenderTarget(ComPtr<IDXGISwapChain> swap_chain) {
    if (!video_player_) {
        std::cerr << "Cannot set render target: video player not initialized" << std::endl;
        return false;
    }
    
    return video_player_->setRenderTarget(swap_chain);
}


bool AudioVideoPlayer::initializeAudio() {
    if (!has_audio_ || !audio_player_) {
        return false;
    }
    
    return audio_player_->initialize();
}


void AudioVideoPlayer::onTimer(double current_time) {
    // 更新视频播放
    if (video_player_) {
        video_player_->onTimer(current_time);
    }
    
    // 更新音频播放
    if (has_audio_ && audio_player_) {
        audio_player_->onTimer(current_time);
    }
}


ComPtr<ID3D11Texture2D> AudioVideoPlayer::getCurrentFrame() const {
    if (!video_player_) {
        return nullptr;
    }
    
    return video_player_->getCurrentFrame();
}


void AudioVideoPlayer::close() {
    // 关闭视频播放器
    if (video_player_) {
        video_player_->close();
    }
    
    // 关闭音频播放器
    if (audio_player_) {
        audio_player_->close();
    }
    
    has_audio_ = false;
    filepath_.clear();
}


bool AudioVideoPlayer::isOpen() const {
    return video_player_ && video_player_->isOpen();
}


bool AudioVideoPlayer::isVideoReady() const {
    return video_player_ && video_player_->isReady();
}


bool AudioVideoPlayer::isAudioReady() const {
    return has_audio_ && audio_player_ && audio_player_->isReady();
}


bool AudioVideoPlayer::hasAudio() const {
    return has_audio_;
}


bool AudioVideoPlayer::isEOF() const {
    // 如果视频到达末尾，则认为播放结束
    bool video_eof = video_player_ && video_player_->isEOF();
    
    // 如果没有音频，只需要检查视频
    if (!has_audio_) {
        return video_eof;
    }
    
    // 如果有音频，检查音频是否也到达末尾
    bool audio_eof = audio_player_ && audio_player_->isEOF();
    
    // 只有音频和视频都到达末尾才认为播放结束
    return video_eof && audio_eof;
}




AudioDecoder* AudioVideoPlayer::getAudioDecoder() const {
    if (!has_audio_ || !audio_player_) {
        return nullptr;
    }
    
    return audio_player_->getAudioDecoder();
}


VideoPlayer* AudioVideoPlayer::getVideoPlayer() const {
    return video_player_.get();
}


AudioPlayer* AudioVideoPlayer::getAudioPlayer() const {
    return audio_player_.get();
}


ID3D11Device* AudioVideoPlayer::getD3D11Device() const {
    if (!video_player_) {
        return nullptr;
    }
    
    return video_player_->getD3D11Device();
}


ID3D11DeviceContext* AudioVideoPlayer::getD3D11DeviceContext() const {
    if (!video_player_) {
        return nullptr;
    }
    
    return video_player_->getD3D11DeviceContext();
}


bool AudioVideoPlayer::getAudioBufferStatus(UINT32& total_frames, UINT32& padding_frames, UINT32& available_frames) const {
    if (!has_audio_ || !audio_player_) {
        return false;
    }
    
    return audio_player_->getAudioBufferStatus(total_frames, padding_frames, available_frames);
} 