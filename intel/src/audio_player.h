#pragma once

#include "audio_decoder.h"
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <atomic>
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

extern "C" {
#include <libavutil/frame.h>
#include <libavcodec/avcodec.h>
}

class AudioPlayer {
public:
	AudioPlayer(); ~AudioPlayer();
	bool open(const std::string& filepath); bool initialize(); void onTimer(double current_time); void close();
	bool isOpen() const; bool isReady() const; bool isEOF() const; AudioDecoder* getAudioDecoder() const; bool getAudioBufferStatus(UINT32& total, UINT32& padding, UINT32& available) const;
private:
	std::unique_ptr<AudioDecoder> audio_decoder_;
	std::queue<AudioDecoder::DecodedFrame> audio_buffer_; mutable std::mutex audio_mutex_;
	MKVStreamReader::StreamInfo stream_info_{}; ComPtr<IMMDevice> audio_device_; ComPtr<IAudioClient> audio_client_; ComPtr<IAudioRenderClient> render_client_;
	UINT32 buffer_frame_count_{}; UINT32 sample_rate_{}; UINT16 audio_channels_{}; UINT16 bits_per_sample_{}; std::atomic<bool> is_audio_initialized_{false}; std::atomic<bool> is_audio_playing_{false};
	void handleAudioFrames(double current_time); void preloadAudioFrames(); double convertAudioPtsToSeconds(AVFrame* frame) const; bool initializeWASAPIAudioPlayer(UINT32 sr, UINT16 ch, UINT16 bps); bool writeAudioData(AVFrame* frame); bool startAudioPlayer(); void stopAudioPlayer(); void closeAudioPlayer(); bool initializeAudioDevice(); bool configureAudioClient(); void cleanupAudioResources(); bool convertFloatToPcm(AVFrame* frame, BYTE* buffer, UINT32 frames_to_write);
};
