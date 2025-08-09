#include "audio_player.h"
#include <iostream>
#include <comdef.h>
#include <algorithm>

extern "C" {
#include <libavutil/rational.h>
#include <libavutil/samplefmt.h>
}

// 初始化 COM 库的 RAII 包装器
class ComInitializer {
public:
    ComInitializer() {
        hr_ = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(hr_)) {
            std::cerr << "Failed to initialize COM library" << std::endl;
        }
    }
    
    ~ComInitializer() {
        if (SUCCEEDED(hr_)) {
            CoUninitialize();
        }
    }
    
    bool isInitialized() const {
        return SUCCEEDED(hr_);
    }
    
private:
    HRESULT hr_;
};

static ComInitializer g_com_initializer;


AudioPlayer::AudioPlayer() 
    : audio_decoder_(std::make_unique<AudioDecoder>())
    , buffer_frame_count_(0)
    , sample_rate_(0)
    , audio_channels_(0)
    , bits_per_sample_(0)
    , is_audio_initialized_(false)
    , is_audio_playing_(false) {
}


AudioPlayer::~AudioPlayer() {
    close();
}


bool AudioPlayer::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    // 打开音频解码器
    if (!audio_decoder_->open(filepath)) {
        std::cerr << "Failed to open audio decoder for: " << filepath << std::endl;
        return false;
    }
    
    // 获取流信息
    MKVStreamReader* stream_reader = audio_decoder_->getStreamReader();
    if (!stream_reader) {
        std::cerr << "Failed to get stream reader from audio decoder" << std::endl;
        close();
        return false;
    }
    
    stream_info_ = stream_reader->getStreamInfo();
    if (stream_info_.audio_stream_index < 0) {
        std::cerr << "No audio stream found" << std::endl;
        close();
        return false;
    }
    
    return true;
}


bool AudioPlayer::initialize() {
    if (!isOpen()) {
        std::cerr << "Cannot initialize: audio not opened" << std::endl;
        return false;
    }
    
    // 初始化 WASAPI 音频播放器
    UINT32 sample_rate = stream_info_.audio_sample_rate;
    UINT16 channels = static_cast<UINT16>(stream_info_.audio_channels);
    UINT16 bits_per_sample = 16;  // 默认16位
    
    if (!initializeWASAPIAudioPlayer(sample_rate, channels, bits_per_sample)) {
        std::cerr << "Failed to initialize WASAPI audio player" << std::endl;
        return false;
    }
    
    // 开始播放
    if (!startAudioPlayer()) {
        std::cerr << "Failed to start WASAPI audio player" << std::endl;
        return false;
    }
    
    return true;
}


void AudioPlayer::onTimer(double current_time) {
    if (!isReady()) {
        return;
    }
    
    handleAudioFrames(current_time);
}


void AudioPlayer::close() {
    std::lock_guard<std::mutex> audio_lock(audio_mutex_);
    
    // 清空音频缓冲
    while (!audio_buffer_.empty()) {
        auto& frame = audio_buffer_.front();
        av_frame_free(&frame.frame);
        audio_buffer_.pop();
    }
    
    // 关闭音频播放器
    closeAudioPlayer();
    
    // 关闭音频解码器
    if (audio_decoder_) {
        audio_decoder_->close();
    }
}


bool AudioPlayer::isOpen() const {
    return audio_decoder_ && audio_decoder_->isOpen();
}


bool AudioPlayer::isReady() const {
    return isOpen() && is_audio_playing_;
}


bool AudioPlayer::isEOF() const {
    return audio_decoder_ && audio_decoder_->isEOF();
}


AudioDecoder* AudioPlayer::getAudioDecoder() const {
    return audio_decoder_.get();
}


bool AudioPlayer::getAudioBufferStatus(UINT32& total_frames, UINT32& padding_frames, UINT32& available_frames) const {
    if (!is_audio_initialized_) {
        return false;
    }
    
    if (!audio_client_) {
        return false;
    }
    
    HRESULT hr = audio_client_->GetCurrentPadding(&padding_frames);
    if (FAILED(hr)) {
        return false;
    }
    
    total_frames = buffer_frame_count_;
    available_frames = buffer_frame_count_ - padding_frames;
    
    return true;
}


void AudioPlayer::handleAudioFrames(double current_time) {
    if (!is_audio_initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> audio_lock(audio_mutex_);
    
    // 预加载音频帧
    preloadAudioFrames();
    
    // 处理音频帧队列
    while (!audio_buffer_.empty()) {
        auto& frame = audio_buffer_.front();
        double frame_time = convertAudioPtsToSeconds(frame.frame);
        
        // 检查音频帧是否到时间（提前100ms推送）
        if (frame_time <= current_time + 0.1) {
            // 尝试写入音频数据
            bool written = writeAudioData(frame.frame);
            
            if (written) {
                // 成功写入，释放帧并移除
                av_frame_free(&frame.frame);
                audio_buffer_.pop();
            } else {
                // 写入失败（缓冲区满），下次再试
                break;
            }
        } else {
            // 时间还没到
            break;
        }
    }
}


void AudioPlayer::preloadAudioFrames() {
    if (!audio_decoder_) {
        return;
    }
    
    // 限制音频缓冲区大小
    const size_t MAX_AUDIO_BUFFER_SIZE = 10;
    
    while (audio_buffer_.size() < MAX_AUDIO_BUFFER_SIZE && !audio_decoder_->isEOF()) {
        AudioDecoder::DecodedFrame frame;
        if (audio_decoder_->readNextFrame(frame)) {
            audio_buffer_.push(frame);
        } else {
            break;
        }
    }
}


double AudioPlayer::convertAudioPtsToSeconds(AVFrame* frame) const {
    if (!frame || frame->pts == AV_NOPTS_VALUE) {
        return 0.0;
    }
    
    // 将 PTS 转换为秒数：pts * time_base.num / time_base.den
    return (double)frame->pts * stream_info_.audio_time_base.num / stream_info_.audio_time_base.den;
}


bool AudioPlayer::initializeWASAPIAudioPlayer(UINT32 sample_rate, UINT16 channels, UINT16 bits_per_sample) {
    if (is_audio_initialized_) {
        closeAudioPlayer();
    }
    
    if (!g_com_initializer.isInitialized()) {
        std::cerr << "COM library not initialized" << std::endl;
        return false;
    }
    
    // 保存参数
    sample_rate_ = sample_rate;
    audio_channels_ = channels;
    bits_per_sample_ = bits_per_sample;
    
    // 初始化音频设备
    if (!initializeAudioDevice()) {
        std::cerr << "Failed to initialize audio device" << std::endl;
        cleanupAudioResources();
        return false;
    }
    
    // 配置音频客户端
    if (!configureAudioClient()) {
        std::cerr << "Failed to configure audio client" << std::endl;
        cleanupAudioResources();
        return false;
    }
    
    is_audio_initialized_ = true;
    return true;
}


bool AudioPlayer::writeAudioData(AVFrame* frame) {
    if (!is_audio_initialized_ || !is_audio_playing_ || !frame) {
        return false;
    }
    
    // 检查缓冲区可用空间
    UINT32 padding_frames;
    HRESULT hr = audio_client_->GetCurrentPadding(&padding_frames);
    if (FAILED(hr)) {
        return false;
    }
    
    UINT32 available_frames = buffer_frame_count_ - padding_frames;
    UINT32 frames_to_write = frame->nb_samples;
    
    // 如果可用空间不足，跳过这次写入（非阻塞）
    if (available_frames < frames_to_write) {
        return false;
    }
    
    // 获取缓冲区
    BYTE* buffer_data;
    hr = render_client_->GetBuffer(frames_to_write, &buffer_data);
    if (FAILED(hr)) {
        return false;
    }
    
    // 转换音频格式并写入缓冲区
    bool success = convertFloatToPcm(frame, buffer_data, frames_to_write);
    
    // 释放缓冲区
    DWORD flags = success ? 0 : AUDCLNT_BUFFERFLAGS_SILENT;
    hr = render_client_->ReleaseBuffer(frames_to_write, flags);
    
    return SUCCEEDED(hr) && success;
}


bool AudioPlayer::startAudioPlayer() {
    if (!is_audio_initialized_ || is_audio_playing_) {
        return false;
    }
    
    HRESULT hr = audio_client_->Start();
    if (SUCCEEDED(hr)) {
        is_audio_playing_ = true;
        return true;
    }
    
    return false;
}


void AudioPlayer::stopAudioPlayer() {
    if (!is_audio_initialized_ || !is_audio_playing_) {
        return;
    }
    
    audio_client_->Stop();
    is_audio_playing_ = false;
}


void AudioPlayer::closeAudioPlayer() {
    if (is_audio_playing_) {
        stopAudioPlayer();
    }
    
    cleanupAudioResources();
    is_audio_initialized_ = false;
}


bool AudioPlayer::initializeAudioDevice() {
    // 创建设备枚举器
    ComPtr<IMMDeviceEnumerator> device_enumerator;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, 
                                  __uuidof(IMMDeviceEnumerator), (void**)&device_enumerator);
    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator" << std::endl;
        return false;
    }
    
    // 获取默认音频渲染设备
    hr = device_enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &audio_device_);
    if (FAILED(hr)) {
        std::cerr << "Failed to get default audio endpoint" << std::endl;
        return false;
    }
    
    // 激活音频客户端
    hr = audio_device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audio_client_);
    if (FAILED(hr)) {
        std::cerr << "Failed to activate audio client" << std::endl;
        return false;
    }
    
    return true;
}


bool AudioPlayer::configureAudioClient() {
    // 获取混音格式
    WAVEFORMATEX* mix_format = nullptr;
    HRESULT hr = audio_client_->GetMixFormat(&mix_format);
    if (FAILED(hr)) {
        std::cerr << "Failed to get mix format" << std::endl;
        return false;
    }
    
    // 设置我们的格式（简化：使用系统混音格式）
    REFERENCE_TIME buffer_duration = 100000;  // 10ms 缓冲区
    hr = audio_client_->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 
                                  buffer_duration, 0, mix_format, nullptr);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client" << std::endl;
        CoTaskMemFree(mix_format);
        return false;
    }
    
    // 获取缓冲区大小
    hr = audio_client_->GetBufferSize(&buffer_frame_count_);
    if (FAILED(hr)) {
        std::cerr << "Failed to get buffer size" << std::endl;
        CoTaskMemFree(mix_format);
        return false;
    }
    
    // 获取渲染客户端
    hr = audio_client_->GetService(__uuidof(IAudioRenderClient), (void**)&render_client_);
    if (FAILED(hr)) {
        std::cerr << "Failed to get render client" << std::endl;
        CoTaskMemFree(mix_format);
        return false;
    }
    
    CoTaskMemFree(mix_format);
    return true;
}


void AudioPlayer::cleanupAudioResources() {
    render_client_.Reset();
    audio_client_.Reset();
    audio_device_.Reset();
    
    buffer_frame_count_ = 0;
    sample_rate_ = 0;
    audio_channels_ = 0;
    bits_per_sample_ = 0;
    is_audio_playing_ = false;
}


bool AudioPlayer::convertFloatToPcm(AVFrame* frame, BYTE* buffer, UINT32 frames_to_write) {
    if (!frame || !buffer) {
        return false;
    }
    
    // 简化实现：假设输入是 AV_SAMPLE_FMT_FLTP（planar float）
    if (frame->format != AV_SAMPLE_FMT_FLTP) {
        std::cerr << "Unsupported audio format: " << frame->format << std::endl;
        return false;
    }
    
    // 转换为 16-bit PCM 立体声
    int16_t* output = (int16_t*)buffer;
    const float* left_channel = (const float*)frame->data[0];
    const float* right_channel = (frame->ch_layout.nb_channels > 1) ? (const float*)frame->data[1] : left_channel;
    
    UINT32 samples_to_convert = (std::min)(frames_to_write, (UINT32)frame->nb_samples);
    
    for (UINT32 i = 0; i < samples_to_convert; i++) {
        // 转换 float [-1.0, 1.0] 到 int16 [-32768, 32767]
        float left_sample = (std::max)(-1.0f, (std::min)(1.0f, left_channel[i]));
        float right_sample = (std::max)(-1.0f, (std::min)(1.0f, right_channel[i]));
        
        output[i * 2] = (int16_t)(left_sample * 32767.0f);
        output[i * 2 + 1] = (int16_t)(right_sample * 32767.0f);
    }
    
    return true;
} 