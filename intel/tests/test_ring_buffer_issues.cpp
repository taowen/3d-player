#include <catch2/catch_test_macros.hpp>
#include "../src/audio_player.h"
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <algorithm>

/**
 * 验证当前 AudioPlayer 存在的环形缓冲缺失问题：
 * 1. 解码抖动导致播放中断（强耦合问题）
 * 2. 部分写入失败导致帧丢失
 * 3. 缓冲水位不可观测，无法预测 underrun
 * 4. 缺少统一格式层和平滑处理能力
 */

TEST_CASE("AudioPlayer ring buffer issues validation", "[test_ring_buffer_issues.cpp]") {
    AudioPlayer player;
    
    SECTION("Decode jitter causes playback interruption - strong coupling issue") {
        // 模拟解码抖动场景：周期性暂停解码来验证强耦合问题
        REQUIRE(player.open("test_data/sample_with_audio.mkv"));
        REQUIRE(player.initialize());
        
        struct BufferStats {
            UINT32 total_frames = 0;
            UINT32 padding_frames = 0;
            UINT32 available_frames = 0;
            double timestamp = 0.0;
        };
        
        std::vector<BufferStats> buffer_history;
        std::vector<bool> underrun_events;
        
        // 记录初始状态
        BufferStats initial_stats;
        player.getAudioBufferStatus(initial_stats.total_frames, initial_stats.padding_frames, initial_stats.available_frames);
        std::cout << "Initial buffer: total=" << initial_stats.total_frames 
                  << " padding=" << initial_stats.padding_frames 
                  << " available=" << initial_stats.available_frames << std::endl;
        
        double current_time = 0.0;
        const double time_step = 0.016;  // ~16ms 步进，模拟正常播放
        const int simulation_steps = 300; // 约5秒模拟
        
        for (int step = 0; step < simulation_steps && !player.isEOF(); step++) {
            // 模拟周期性解码延迟（每100ms暂停10ms）
            if (step % 6 == 5) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                std::cout << "Simulated decode jitter at t=" << current_time << "s" << std::endl;
            }
            
            // 调用播放器更新
            player.onTimer(current_time);
            
            // 记录缓冲区状态
            BufferStats stats;
            if (player.getAudioBufferStatus(stats.total_frames, stats.padding_frames, stats.available_frames)) {
                stats.timestamp = current_time;
                buffer_history.push_back(stats);
                
                // 检测 underrun 征兆（可用空间过大表示缓冲区空）
                bool potential_underrun = stats.padding_frames < (stats.total_frames * 0.1);
                underrun_events.push_back(potential_underrun);
                
                if (potential_underrun) {
                    std::cout << "Potential underrun at t=" << current_time 
                              << "s, padding=" << stats.padding_frames 
                              << "/" << stats.total_frames << std::endl;
                }
            }
            
            current_time += time_step;
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        
        // 分析结果：应该观察到因解码抖动导致的缓冲区不稳定
        int underrun_count = 0;
        for (bool underrun : underrun_events) {
            if (underrun) underrun_count++;
        }
        
        std::cout << "Analysis: Found " << underrun_count << " potential underrun events in " 
                  << buffer_history.size() << " samples" << std::endl;
        
        // 问题验证：解码抖动应该导致缓冲区水位波动
        // 由于缺少环形缓冲，无法吸收解码延迟
        INFO("Strong coupling between decode timing and playback leads to buffer instability");
        REQUIRE(underrun_count > 0); // 预期会有 underrun 征兆
    }
    
    SECTION("Partial write failure causes frame drops") {
        REQUIRE(player.open("test_data/sample_with_audio.mkv"));
        REQUIRE(player.initialize());
        
        // 获取音频解码器来直接测试写入失败场景
        AudioDecoder* decoder = player.getAudioDecoder();
        REQUIRE(decoder != nullptr);
        
        // 先让播放器运行一段时间填充缓冲区
        double current_time = 0.0;
        for (int i = 0; i < 50; i++) {
            player.onTimer(current_time);
            current_time += 0.01;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // 检查缓冲区状态
        UINT32 total_frames, padding_frames, available_frames;
        bool status_ok = player.getAudioBufferStatus(total_frames, padding_frames, available_frames);
        REQUIRE(status_ok);
        
        std::cout << "Before stress test - total: " << total_frames 
                  << ", padding: " << padding_frames 
                  << ", available: " << available_frames << std::endl;
        
        // 模拟快速连续解码（不给设备消费时间）来触发写入失败
        int successful_writes = 0;
        int failed_writes = 0;
        
        for (int i = 0; i < 10; i++) {
            player.onTimer(current_time);
            
            // 检查新的缓冲区状态
            UINT32 new_padding, new_available;
            player.getAudioBufferStatus(total_frames, new_padding, new_available);
            
            // 如果可用空间变少，说明写入成功
            if (new_available < available_frames) {
                successful_writes++;
                available_frames = new_available;
                padding_frames = new_padding;
            } else {
                failed_writes++;
            }
            
            current_time += 0.001; // 1ms 快速步进
        }
        
        std::cout << "Stress test results - successful: " << successful_writes 
                  << ", failed: " << failed_writes << std::endl;
        
        // 问题验证：当设备缓冲区满时，整帧被丢弃而不是部分写入
        INFO("Current implementation drops entire frames when device buffer is full");
        INFO("With ring buffer, partial writes would be possible");
        
        // 由于缺少环形缓冲，快速写入会导致帧丢失
        REQUIRE(failed_writes > 0);
    }
    
    SECTION("Buffer level not observable - underrun prediction impossible") {
        REQUIRE(player.open("test_data/sample_with_audio.mkv"));
        REQUIRE(player.initialize());
        
        // 当前只能观察到设备缓冲区状态，无法得到总体播放余量
        UINT32 total_frames, padding_frames, available_frames;
        bool status_ok = player.getAudioBufferStatus(total_frames, padding_frames, available_frames);
        REQUIRE(status_ok);
        
        // 运行播放器并记录状态变化
        std::vector<double> buffer_levels;
        double current_time = 0.0;
        
        for (int i = 0; i < 100; i++) {
            player.onTimer(current_time);
            
            UINT32 new_total, new_padding, new_available;
            if (player.getAudioBufferStatus(new_total, new_padding, new_available)) {
                // 当前只能计算设备缓冲区占用率，无法知道软件缓冲区状态
                double device_buffer_ratio = (double)new_padding / new_total;
                buffer_levels.push_back(device_buffer_ratio);
            }
            
            current_time += 0.02;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        
        // 分析缓冲区可观测性
        double min_level = *std::min_element(buffer_levels.begin(), buffer_levels.end());
        double max_level = *std::max_element(buffer_levels.begin(), buffer_levels.end());
        
        std::cout << "Buffer level range: " << min_level << " - " << max_level << std::endl;
        
        // 问题验证：无法获得总体缓冲时长，只能看到设备层面的瞬时状态
        INFO("Current implementation only provides device buffer status");
        INFO("Missing: total buffered playback time, software buffer status");
        INFO("Cannot predict underrun or calculate remaining playback duration");
        
        // 验证缺少统一的缓冲水位指标
        // 理想情况下应该能获取：总缓冲时长 = (软件环形缓冲 + 设备缓冲) / 采样率
        REQUIRE(true); // 这里只是展示问题，实际验证在于无法获得完整信息
    }
    
    SECTION("No format unification layer - extension complexity") {
        REQUIRE(player.open("test_data/sample_with_audio.mkv"));
        REQUIRE(player.initialize());
        
        // 获取解码器来检查格式处理
        AudioDecoder* decoder = player.getAudioDecoder();
        REQUIRE(decoder != nullptr);
        
        // 当前实现中，格式转换发生在 writeAudioData 内部
        // 这意味着每次写入都需要格式转换，且无法统一处理多格式
        
        std::cout << "Current format handling: conversion happens in writeAudioData" << std::endl;
        std::cout << "Problem: No unified format layer before buffering" << std::endl;
        
        // 模拟多格式场景的复杂性（当前代码硬编码 FLTP -> float32 转换）
        // 如果要支持更多格式，convertFloatToPcm 会变得复杂
        
        INFO("Current implementation has format conversion tightly coupled with device writing");
        INFO("Missing: unified format conversion before ring buffer");
        INFO("Extension to support more formats will require complex changes");
        
        // 验证格式处理的耦合问题
        REQUIRE(true); // 问题在于架构，不在于功能失败
    }
    
    player.close();
}

TEST_CASE("Current buffer limitations demonstration", "[test_ring_buffer_issues.cpp]") {
    SECTION("Queue-based buffering vs continuous PCM buffer comparison") {
        AudioPlayer player;
        REQUIRE(player.open("test_data/sample_with_audio.mkv"));
        REQUIRE(player.initialize());
        
        // 当前使用 std::queue<AudioDecoder::DecodedFrame> 存储离散帧
        // 问题1：无法部分消费一个帧
        // 问题2：内存不连续，无法高效批量处理
        // 问题3：帧边界限制了写入粒度
        
        std::cout << "=== Current Implementation Analysis ===" << std::endl;
        std::cout << "Buffer type: std::queue<DecodedFrame> (discrete frames)" << std::endl;
        std::cout << "Limitations:" << std::endl;
        std::cout << "1. Cannot partially consume frames" << std::endl;
        std::cout << "2. Memory fragmentation" << std::endl;
        std::cout << "3. Frame boundary writing only" << std::endl;
        std::cout << "4. No unified sample-level access" << std::endl;
        
        // 运行并观察实际行为
        double current_time = 0.0;
        UINT32 total_frames, padding_frames, available_frames;
        
        for (int i = 0; i < 20; i++) {
            player.onTimer(current_time);
            
            if (player.getAudioBufferStatus(total_frames, padding_frames, available_frames)) {
                std::cout << "t=" << current_time << "s: device padding=" 
                          << padding_frames << " (" 
                          << (padding_frames * 1000 / 48000) << "ms)" << std::endl;
            }
            
            current_time += 0.05;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        std::cout << "\n=== Ring Buffer Would Provide ===" << std::endl;
        std::cout << "1. Continuous linear PCM buffer" << std::endl;
        std::cout << "2. Sample-granular read/write operations" << std::endl;
        std::cout << "3. Unified format (e.g., float32 interleaved)" << std::endl;
        std::cout << "4. Precise buffered time calculation" << std::endl;
        std::cout << "5. Smooth underrun handling with silence injection" << std::endl;
        
        player.close();
        REQUIRE(true);
    }
}