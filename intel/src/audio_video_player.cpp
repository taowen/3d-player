#include "audio_video_player.h"
#include <iostream>
AudioVideoPlayer::AudioVideoPlayer():video_player_(std::make_unique<VideoPlayer>()),audio_player_(std::make_unique<AudioPlayer>()){}
AudioVideoPlayer::~AudioVideoPlayer(){ close(); }
bool AudioVideoPlayer::open(const std::string& fp){ if(isOpen()) close(); filepath_=fp; if(!video_player_->open(fp)){ std::cerr<<"Video open failed"<<std::endl; return false;} has_audio_=audio_player_->open(fp); if(!has_audio_) std::cerr<<"No audio stream or open failed"<<std::endl; return true; }
bool AudioVideoPlayer::setRenderTarget(ComPtr<ID3D11Texture2D> rt){ return video_player_->setRenderTarget(rt); }
bool AudioVideoPlayer::setRenderTarget(ComPtr<IDXGISwapChain> sc){ return video_player_->setRenderTarget(sc); }
bool AudioVideoPlayer::initializeAudio(){ if(!has_audio_) return false; return audio_player_->initialize(); }
void AudioVideoPlayer::onTimer(double t){ if(video_player_) video_player_->onTimer(t); if(has_audio_ && audio_player_) audio_player_->onTimer(t); }
ComPtr<ID3D11Texture2D> AudioVideoPlayer::getCurrentFrame() const { return video_player_?video_player_->getCurrentFrame():nullptr; }
void AudioVideoPlayer::close(){ if(video_player_) video_player_->close(); if(audio_player_) audio_player_->close(); has_audio_=false; filepath_.clear(); }
bool AudioVideoPlayer::isOpen() const { return video_player_ && video_player_->isOpen(); }
bool AudioVideoPlayer::isVideoReady() const { return video_player_ && video_player_->isReady(); }
bool AudioVideoPlayer::isAudioReady() const { return has_audio_ && audio_player_ && audio_player_->isReady(); }
bool AudioVideoPlayer::hasAudio() const { return has_audio_; }
bool AudioVideoPlayer::isEOF() const { bool ve = video_player_ && video_player_->isEOF(); if(!has_audio_) return ve; bool ae = audio_player_ && audio_player_->isEOF(); return ve && ae; }
AudioDecoder* AudioVideoPlayer::getAudioDecoder() const { return has_audio_?audio_player_->getAudioDecoder():nullptr; }
VideoPlayer* AudioVideoPlayer::getVideoPlayer() const { return video_player_.get(); }
AudioPlayer* AudioVideoPlayer::getAudioPlayer() const { return audio_player_.get(); }
ID3D11Device* AudioVideoPlayer::getD3D11Device() const { return video_player_?video_player_->getD3D11Device():nullptr; }
ID3D11DeviceContext* AudioVideoPlayer::getD3D11DeviceContext() const { return video_player_?video_player_->getD3D11DeviceContext():nullptr; }
bool AudioVideoPlayer::getAudioBufferStatus(UINT32& total, UINT32& padding, UINT32& available) const { if(!has_audio_||!audio_player_) return false; return audio_player_->getAudioBufferStatus(total,padding,available); }
