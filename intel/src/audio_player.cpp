// Local simplified audio player implementation (Intel variant)
#include "audio_player.h"
#include <iostream>

extern "C" {
#include <libavutil/frame.h>
#include <libavcodec/avcodec.h>
}

AudioPlayer::AudioPlayer() : audio_decoder_(std::make_unique<AudioDecoder>()) {}
AudioPlayer::~AudioPlayer() { close(); }

bool AudioPlayer::open(const std::string& filepath) {
	if (isOpen()) close();
	if (!audio_decoder_->open(filepath)) {
		std::cerr << "Audio decoder open failed" << std::endl;
		return false;
	}
	stream_info_ = audio_decoder_->getStreamReader()->getStreamInfo();
	return true;
}

bool AudioPlayer::initialize() {
	if (!isOpen()) return false;
	if (is_audio_initialized_) return true;
	AVCodecParameters* cp = audio_decoder_->getStreamReader()->getAudioCodecParameters();
	if (!cp) return false;
	sample_rate_ = cp->sample_rate;
	audio_channels_ = (cp->ch_layout.nb_channels > 0 ? cp->ch_layout.nb_channels : 2);
	bits_per_sample_ = 16;
	if (!initializeWASAPIAudioPlayer(sample_rate_, (UINT16)audio_channels_, bits_per_sample_)) return false;
	is_audio_initialized_ = true;
	preloadAudioFrames();
	return true;
}

void AudioPlayer::onTimer(double current_time) {
	if (!is_audio_initialized_) return;
	handleAudioFrames(current_time);
}

void AudioPlayer::close() {
	std::lock_guard<std::mutex> lk(audio_mutex_);
	while (!audio_buffer_.empty()) {
		auto& f = audio_buffer_.front();
		if (f.frame) av_frame_free(&f.frame);
		audio_buffer_.pop();
	}
	if (audio_decoder_) audio_decoder_->close();
	stopAudioPlayer();
	closeAudioPlayer();
	cleanupAudioResources();
	is_audio_initialized_ = false;
	is_audio_playing_ = false;
}

bool AudioPlayer::isOpen() const { return audio_decoder_ && audio_decoder_->isOpen(); }
bool AudioPlayer::isReady() const { return isOpen() && is_audio_initialized_; }
bool AudioPlayer::isEOF() const { return audio_decoder_ && audio_decoder_->isEOF(); }
AudioDecoder* AudioPlayer::getAudioDecoder() const { return audio_decoder_.get(); }

bool AudioPlayer::getAudioBufferStatus(UINT32& total, UINT32& padding, UINT32& available) const {
	if (!audio_client_) return false;
	if (FAILED(audio_client_->GetBufferSize(&total))) return false;
	if (FAILED(audio_client_->GetCurrentPadding(&padding))) return false;
	available = total - padding;
	return true;
}

void AudioPlayer::handleAudioFrames(double /*current_time*/) {
	preloadAudioFrames();
	if (!is_audio_playing_) {
		if (!startAudioPlayer()) return;
	}
	while (!audio_buffer_.empty()) {
		UINT32 total = 0, padding = 0, available = 0;
		if (!getAudioBufferStatus(total, padding, available) || available == 0) break;
		auto& front = audio_buffer_.front();
		if (!front.is_valid || !front.frame) {
			audio_buffer_.pop();
			continue;
		}
		UINT32 frames_to_write = (UINT32)front.frame->nb_samples;
		if (frames_to_write > available) break;
		if (!writeAudioData(front.frame)) break;
		av_frame_free(&front.frame);
		audio_buffer_.pop();
	}
}

void AudioPlayer::preloadAudioFrames() {
	std::lock_guard<std::mutex> lk(audio_mutex_);
	if (audio_buffer_.size() >= 5) return;
	while (audio_buffer_.size() < 5 && !audio_decoder_->isEOF()) {
		AudioDecoder::DecodedFrame f;
		if (!audio_decoder_->readNextFrame(f)) break;
		if (f.is_valid) audio_buffer_.push(f); else break;
	}
}

double AudioPlayer::convertAudioPtsToSeconds(AVFrame* frame) const {
	if (!frame || frame->pts == AV_NOPTS_VALUE) return 0.0;
	AVRational tb = audio_decoder_->getStreamReader()->getStreamInfo().audio_time_base;
	return frame->pts * av_q2d(tb);
}

bool AudioPlayer::initializeWASAPIAudioPlayer(UINT32 sr, UINT16 ch, UINT16 bps) {
	if (!initializeAudioDevice()) return false;
	if (!configureAudioClient()) return false;
	if (FAILED(audio_client_->GetBufferSize(&buffer_frame_count_))) return false;
	if (FAILED(audio_client_->GetService(__uuidof(IAudioRenderClient), (void**)&render_client_))) return false;
	return true;
}

bool AudioPlayer::writeAudioData(AVFrame* frame) {
	if (!frame || !render_client_) return false;
	const int bytes_per_sample = bits_per_sample_ / 8;
	const UINT32 frames = frame->nb_samples;
	BYTE* data = nullptr;
	if (FAILED(render_client_->GetBuffer(frames, &data))) return false;
	if (frame->format == AV_SAMPLE_FMT_FLTP) {
		for (int i = 0; i < (int)frames; ++i) {
			for (int c = 0; c < audio_channels_; ++c) {
				float sample = ((float*)frame->data[c])[i];
				if (sample > 1.0f) sample = 1.0f; else if (sample < -1.0f) sample = -1.0f;
				int16_t s = (int16_t)(sample * 32767.0f);
				*(int16_t*)(data + (i * audio_channels_ + c) * bytes_per_sample) = s;
			}
		}
	} else if (frame->format == AV_SAMPLE_FMT_FLT) {
		const float* in = (const float*)frame->data[0];
		for (int i = 0; i < (int)(frames * audio_channels_); ++i) {
			float sample = in[i];
			if (sample > 1.0f) sample = 1.0f; else if (sample < -1.0f) sample = -1.0f;
			int16_t s = (int16_t)(sample * 32767.0f);
			((int16_t*)data)[i] = s;
		}
	} else if (frame->format == AV_SAMPLE_FMT_S16) {
		memcpy(data, frame->data[0], frames * audio_channels_ * bytes_per_sample);
	} else {
		memset(data, 0, frames * audio_channels_ * bytes_per_sample);
	}
	if (FAILED(render_client_->ReleaseBuffer(frames, 0))) return false;
	return true;
}

bool AudioPlayer::startAudioPlayer() {
	if (is_audio_playing_) return true;
	if (!audio_client_) return false;
	if (FAILED(audio_client_->Start())) return false;
	is_audio_playing_ = true;
	return true;
}

void AudioPlayer::stopAudioPlayer() {
	if (!is_audio_playing_) return;
	if (audio_client_) audio_client_->Stop();
	is_audio_playing_ = false;
}

void AudioPlayer::closeAudioPlayer() { stopAudioPlayer(); }

bool AudioPlayer::initializeAudioDevice() {
	ComPtr<IMMDeviceEnumerator> enumerator;
	if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&enumerator)))) return false;
	if (FAILED(enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &audio_device_))) return false;
	return true;
}

bool AudioPlayer::configureAudioClient() {
	if (!audio_device_) return false;
	if (FAILED(audio_device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audio_client_))) return false;
	WAVEFORMATEX* pwfx = nullptr;
	if (FAILED(audio_client_->GetMixFormat(&pwfx))) return false;
	REFERENCE_TIME bufferDuration = 10000000; // 1s
	if (FAILED(audio_client_->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, bufferDuration, 0, pwfx, nullptr))) { CoTaskMemFree(pwfx); return false; }
	CoTaskMemFree(pwfx);
	return true;
}

void AudioPlayer::cleanupAudioResources() {
	render_client_.Reset();
	audio_client_.Reset();
	audio_device_.Reset();
}

bool AudioPlayer::convertFloatToPcm(AVFrame*, BYTE*, UINT32) { return true; }
