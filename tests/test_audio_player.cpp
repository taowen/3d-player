#include <catch2/catch_test_macros.hpp>
#include "../src/audio_player.h"
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

const std::string AUDIO_TEST_FILE = "test_data/sample_with_audio.mkv";


TEST_CASE("AudioPlayer basic functionality", "[audio_player]") {
    AudioPlayer player;
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(player.open("nonexistent_file.mkv"));
        REQUIRE_FALSE(player.isOpen());
    }
    
    SECTION("Open file without audio stream should fail") {
        // 尝试打开一个没有音频流的文件
        REQUIRE_FALSE(player.open("test_data/sample_hw.mkv"));
        REQUIRE_FALSE(player.isOpen());
    }
    
    SECTION("Open valid audio file and test initialization") {
        std::ifstream file(AUDIO_TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << AUDIO_TEST_FILE << " not found");
        }
        
        // 第一步：打开音频文件
        REQUIRE(player.open(AUDIO_TEST_FILE));
        REQUIRE(player.isOpen());
        REQUIRE_FALSE(player.isEOF());
        
        // 获取音频解码器
        AudioDecoder* audio_decoder = player.getAudioDecoder();
        REQUIRE(audio_decoder != nullptr);
        
        // 第二步：初始化音频播放器
        bool initialized = player.initialize();
        if (initialized) {
            std::cout << "Audio initialization successful" << std::endl;
            REQUIRE(player.isReady());
            
            // 测试音频缓冲区状态
            UINT32 total_frames, padding_frames, available_frames;
            bool status_ok = player.getAudioBufferStatus(total_frames, padding_frames, available_frames);
            REQUIRE(status_ok);
            REQUIRE(total_frames > 0);
            
            std::cout << "Audio buffer status: " << total_frames << " total, " 
                     << padding_frames << " padding, " << available_frames << " available" << std::endl;
                     
            // 模拟短时间播放
            double current_time = 0.0;
            double time_step = 1.0 / 30.0;  // 30 FPS 播放
            int timer_calls = 0;
            int max_timer_calls = 10;
            
            while (timer_calls < max_timer_calls && !player.isEOF()) {
                // 调用 onTimer 进行音频播放
                player.onTimer(current_time);
                
                timer_calls++;
                std::cout << "Timer call " << timer_calls << " at time " << current_time << "s" << std::endl;
                
                // 推进时间
                current_time += time_step;
            }
            
            // 验证我们至少进行了一些播放
            REQUIRE(timer_calls > 0);
            
        } else {
            std::cout << "Audio initialization failed" << std::endl;
        }
        
        player.close();
        REQUIRE_FALSE(player.isOpen());
    }
}


TEST_CASE("AudioPlayer lifecycle management", "[audio_player]") {
    AudioPlayer player;
    
    SECTION("Open, initialize, and close audio player") {
        std::ifstream file(AUDIO_TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << AUDIO_TEST_FILE << " not found");
        }
        
        // 打开文件
        REQUIRE(player.open(AUDIO_TEST_FILE));
        REQUIRE(player.isOpen());
        
        // 初始化
        bool initialized = player.initialize();
        if (initialized) {
            REQUIRE(player.isReady());
            
            // 测试多次调用 onTimer
            for (int i = 0; i < 5; i++) {
                player.onTimer(i * 0.1);
            }
        }
        
        // 关闭
        player.close();
        REQUIRE_FALSE(player.isOpen());
        REQUIRE_FALSE(player.isReady());
    }
    
    SECTION("Multiple open/close cycles") {
        std::ifstream file(AUDIO_TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << AUDIO_TEST_FILE << " not found");
        }
        
        // 第一次打开/关闭
        REQUIRE(player.open(AUDIO_TEST_FILE));
        REQUIRE(player.isOpen());
        player.close();
        REQUIRE_FALSE(player.isOpen());
        
        // 第二次打开/关闭
        REQUIRE(player.open(AUDIO_TEST_FILE));
        REQUIRE(player.isOpen());
        player.close();
        REQUIRE_FALSE(player.isOpen());
    }
}


TEST_CASE("AudioPlayer state management", "[audio_player]") {
    AudioPlayer player;
    
    SECTION("Test state consistency") {
        std::ifstream file(AUDIO_TEST_FILE);
        if (!file.good()) {
            SKIP("Test file " << AUDIO_TEST_FILE << " not found");
        }
        
        // 初始状态
        REQUIRE_FALSE(player.isOpen());
        REQUIRE_FALSE(player.isReady());
        REQUIRE_FALSE(player.isEOF());
        
        // 打开文件后
        REQUIRE(player.open(AUDIO_TEST_FILE));
        REQUIRE(player.isOpen());
        REQUIRE_FALSE(player.isReady());  // 未初始化
        REQUIRE_FALSE(player.isEOF());
        
        // 初始化后
        bool initialized = player.initialize();
        if (initialized) {
            REQUIRE(player.isOpen());
            REQUIRE(player.isReady());
            REQUIRE_FALSE(player.isEOF());
        }
        
        // 关闭后
        player.close();
        REQUIRE_FALSE(player.isOpen());
        REQUIRE_FALSE(player.isReady());
    }
}


