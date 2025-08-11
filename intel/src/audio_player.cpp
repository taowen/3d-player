#include "audio_player.h"
#include <iostream>
#include <comdef.h>
#include <algorithm>
#include <vector>
#include <cstring>

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
    , ring_buffer_(nullptr)
    , format_converter_(std::make_unique<AudioFormatConverter>())
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
    
    // 配置格式转换器
    AVCodecContext* codec_ctx = audio_decoder_->getAVCodecContext();
    if (!codec_ctx) {
        std::cerr << "Failed to get codec context" << std::endl;
        return false;
    }
    
    if (!format_converter_->configure(codec_ctx->sample_rate, codec_ctx->sample_fmt, codec_ctx->ch_layout.nb_channels)) {
        std::cerr << "Failed to configure format converter" << std::endl;
        return false;
    }
    
    // 创建环形缓冲区（500ms缓冲）
    const size_t buffer_duration_samples = format_converter_->output_sample_rate() / 2; // 500ms
    ring_buffer_ = std::make_unique<RingBuffer>(buffer_duration_samples, format_converter_->output_channels());
    
    // 初始化 WASAPI 音频播放器
    UINT32 sample_rate = format_converter_->output_sample_rate();
    UINT16 channels = static_cast<UINT16>(format_converter_->output_channels());
    UINT16 bits_per_sample = 32;  // float32 = 32位
    
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
    (void)current_time; // 暂未使用，但保留接口一致性
    
    if (!isReady()) {
        return;
    }
    
    handleAudioFrames();
}


void AudioPlayer::close() {
    std::lock_guard<std::mutex> audio_lock(audio_mutex_);
    
    // 清空环形缓冲区
    if (ring_buffer_) {
        ring_buffer_->clear();
        ring_buffer_.reset();
    }
    
    // 重置格式转换器
    format_converter_.reset();
    format_converter_ = std::make_unique<AudioFormatConverter>();
    
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


void AudioPlayer::handleAudioFrames() {
    if (!is_audio_initialized_ || !ring_buffer_ || !format_converter_) {
        return;
    }
    
    std::lock_guard<std::mutex> audio_lock(audio_mutex_);
    
    // 解码并写入环形缓冲区
    fillRingBuffer();
    
    // 从环形缓冲区写入到WASAPI
    writeToWASAPI();
}


void AudioPlayer::fillRingBuffer() {
    if (!audio_decoder_ || !ring_buffer_ || !format_converter_) {
        return;
    }
    
    // 保持环形缓冲区有足够的数据（至少300ms）
    const size_t min_samples = format_converter_->output_sample_rate() * 300 / 1000; // 300ms
    
    while (ring_buffer_->available_for_read() < min_samples && !audio_decoder_->isEOF()) {
        AudioDecoder::DecodedFrame frame;
        if (!audio_decoder_->readNextFrame(frame)) {
            break;
        }
        
        // 格式转换
        float* converted_samples = nullptr;
        int converted_sample_count = 0;
        
        int ret = format_converter_->convert(frame.frame, &converted_samples, &converted_sample_count);
        
        if (ret == 0 && converted_samples && converted_sample_count > 0) {
            // 写入环形缓冲区
            size_t samples_written = ring_buffer_->write(converted_samples, converted_sample_count);
            
            if (samples_written < static_cast<size_t>(converted_sample_count)) {
                // 环形缓冲区满，稍后再试
                av_frame_free(&frame.frame);
                break;
            }
        }
        
        av_frame_free(&frame.frame);
    }
}


void AudioPlayer::writeToWASAPI() {
    if (!is_audio_initialized_ || !is_audio_playing_ || !ring_buffer_) {
        return;
    }
    
    // 检查缓冲区可用空间
    UINT32 padding_frames;
    HRESULT hr = audio_client_->GetCurrentPadding(&padding_frames);
    if (FAILED(hr)) {
        return;
    }
    
    UINT32 available_frames = buffer_frame_count_ - padding_frames;
    if (available_frames == 0) {
        return;
    }
    
    // 计算要写入的帧数
    size_t ring_available_samples = ring_buffer_->available_for_read();
    UINT32 frames_to_write = (std::min)(available_frames, static_cast<UINT32>(ring_available_samples));
    
    if (frames_to_write == 0) {
        // 环形缓冲区空，填充静音防止 underrun
        fillSilence(available_frames);
        return;
    }
    
    // 获取缓冲区
    BYTE* buffer_data;
    hr = render_client_->GetBuffer(frames_to_write, &buffer_data);
    if (FAILED(hr)) {
        return;
    }
    
    // 从环形缓冲区读取数据
    std::vector<float> temp_buffer(frames_to_write * ring_buffer_->channels());
    size_t samples_read = ring_buffer_->read(temp_buffer.data(), frames_to_write);
    
    if (samples_read > 0) {
        // 写入到WASAPI缓冲区 (float32 -> float32, 直接拷贝)
        std::memcpy(buffer_data, temp_buffer.data(), samples_read * ring_buffer_->channels() * sizeof(float));
    }
    
    // 释放缓冲区
    DWORD flags = (samples_read > 0) ? 0 : AUDCLNT_BUFFERFLAGS_SILENT;
    hr = render_client_->ReleaseBuffer(static_cast<UINT32>(samples_read), flags);
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
    
    // 转换为 float32 立体声 (WASAPI mix format)
    float* output = (float*)buffer;
    const float* left_channel = (const float*)frame->data[0];
    const float* right_channel = (frame->ch_layout.nb_channels > 1) ? (const float*)frame->data[1] : left_channel;
    
    UINT32 samples_to_convert = (std::min)(frames_to_write, (UINT32)frame->nb_samples);
    
    for (UINT32 i = 0; i < samples_to_convert; i++) {
        // 直接复制 float [-1.0, 1.0] 数据
        float left_sample = (std::max)(-1.0f, (std::min)(1.0f, left_channel[i]));
        float right_sample = (std::max)(-1.0f, (std::min)(1.0f, right_channel[i]));
        
        output[i * 2] = left_sample;
        output[i * 2 + 1] = right_sample;
    }
    
    return true;
}




void AudioPlayer::fillSilence(UINT32 frame_count) {
    if (!is_audio_initialized_ || !render_client_ || frame_count == 0) {
        return;
    }
    
    // 获取缓冲区
    BYTE* buffer_data;
    HRESULT hr = render_client_->GetBuffer(frame_count, &buffer_data);
    if (FAILED(hr)) {
        return;
    }
    
    // 填充静音 (float32格式下为0.0)
    std::memset(buffer_data, 0, frame_count * audio_channels_ * sizeof(float));
    
    // 释放缓冲区并标记为静音
    hr = render_client_->ReleaseBuffer(frame_count, AUDCLNT_BUFFERFLAGS_SILENT);
} 