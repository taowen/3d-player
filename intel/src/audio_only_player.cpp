#include "audio_player.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <windows.h>

class AudioOnlyPlayer {
public:
    AudioOnlyPlayer() : start_time_(std::chrono::steady_clock::now()), is_playing_(false) {}
    
    bool open(const std::string& filepath) {
        if (!audio_player_.open(filepath)) {
            std::cerr << "Failed to open audio file: " << filepath << std::endl;
            return false;
        }
        
        if (!audio_player_.initialize()) {
            std::cerr << "Failed to initialize audio player" << std::endl;
            return false;
        }
        
        std::cout << "Audio file opened successfully: " << filepath << std::endl;
        
        // Get stream info
        auto* decoder = audio_player_.getAudioDecoder();
        if (decoder && decoder->getStreamReader()) {
            auto stream_info = decoder->getStreamReader()->getStreamInfo();
            std::cout << "Audio codec: " << stream_info.audio_codec << std::endl;
            std::cout << "Sample rate: " << stream_info.audio_sample_rate << " Hz" << std::endl;
            std::cout << "Channels: " << stream_info.audio_channels << std::endl;
            std::cout << "Duration: " << stream_info.duration << " seconds" << std::endl;
            std::cout << "Audio bitrate: " << stream_info.audio_bitrate << " bps" << std::endl;
        }
        
        return true;
    }
    
    void play() {
        if (!audio_player_.isReady()) {
            std::cerr << "Audio player is not ready" << std::endl;
            return;
        }
        
        is_playing_ = true;
        start_time_ = std::chrono::steady_clock::now();
        
        std::cout << "Starting audio playback..." << std::endl;
        std::cout << "Press ESC to exit" << std::endl;
        
        while (is_playing_ && !audio_player_.isEOF()) {
            auto current_time_point = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                current_time_point - start_time_);
            double current_time = elapsed.count() / 1000000.0;
            
            audio_player_.onTimer(current_time);
            
            // Check for ESC key
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
                std::cout << "\nPlayback stopped by user" << std::endl;
                break;
            }
            
            // Update every 10ms
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Print progress every second
            static int last_second = -1;
            int current_second = static_cast<int>(current_time);
            if (current_second != last_second) {
                last_second = current_second;
                std::cout << "\rPlayback time: " << current_second << "s" << std::flush;
            }
        }
        
        if (audio_player_.isEOF()) {
            std::cout << "\nPlayback completed" << std::endl;
        }
        
        is_playing_ = false;
    }
    
    void close() {
        audio_player_.close();
    }

private:
    AudioPlayer audio_player_;
    std::chrono::steady_clock::time_point start_time_;
    bool is_playing_;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <mkv_file_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " test_data/sample_with_audio.mkv" << std::endl;
        return 1;
    }
    
    std::string filepath = argv[1];
    
    // Initialize COM for WASAPI
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM: " << std::hex << hr << std::endl;
        return 1;
    }
    
    AudioOnlyPlayer player;
    
    if (!player.open(filepath)) {
        CoUninitialize();
        return 1;
    }
    
    player.play();
    player.close();
    
    CoUninitialize();
    return 0;
}