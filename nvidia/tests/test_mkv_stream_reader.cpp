#include <catch2/catch_test_macros.hpp>
#include "../src/mkv_stream_reader.h"
#include <iostream>
#include <fstream>

// Test MKV file path - use hardware decoder friendly test file
const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

TEST_CASE("MKVStreamReader can open and read stream info", "[mkv_stream_reader][test_mkv_stream_reader.cpp]") {
    MKVStreamReader reader;
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(reader.open("non_existent_file.mkv"));
        REQUIRE_FALSE(reader.isOpen());
    }
    
    SECTION("Open valid MKV file should succeed") {
        // Skip test if file doesn't exist
        std::ifstream file(TEST_MKV_FILE);
        if (!file.good()) {
            SKIP("Test file " << TEST_MKV_FILE << " not found");
        }
        
        // Check if test file exists
        REQUIRE(reader.open(TEST_MKV_FILE));
        REQUIRE(reader.isOpen());
        REQUIRE_FALSE(reader.isEOF());
        
        reader.close();
        REQUIRE_FALSE(reader.isOpen());
    }
    
    SECTION("Stream info extraction") {
        // Get stream information
        std::ifstream file(TEST_MKV_FILE);
        if (!file.good()) {
            SKIP("Test file " << TEST_MKV_FILE << " not found");
        }
        
        // Verify basic stream information
        REQUIRE(reader.open(TEST_MKV_FILE));
        
        auto info = reader.getStreamInfo();
        
        // Should have at least video stream
        REQUIRE(info.video_stream_index >= 0);
        REQUIRE(info.width > 0);
        REQUIRE(info.height > 0);
        REQUIRE(info.duration > 0.0);
        REQUIRE_FALSE(info.video_codec.empty());
        
        // Time base should be valid
        REQUIRE(info.video_time_base.num > 0);
        REQUIRE(info.video_time_base.den > 0);
        
        // Print info for debugging
        std::cout << "Video codec: " << info.video_codec << std::endl;
        std::cout << "Resolution: " << info.width << "x" << info.height << std::endl;
        std::cout << "Duration: " << info.duration << " seconds" << std::endl;
        std::cout << "FPS: " << info.fps << std::endl;
        std::cout << "Video bitrate: " << info.video_bitrate << std::endl;
        
        if (info.audio_stream_index >= 0) {
            std::cout << "Audio codec: " << info.audio_codec << std::endl;
            std::cout << "Sample rate: " << info.audio_sample_rate << std::endl;
            std::cout << "Channels: " << info.audio_channels << std::endl;
            std::cout << "Audio bitrate: " << info.audio_bitrate << std::endl;
        }
        
        reader.close();
    }
}

TEST_CASE("MKVStreamReader packet reading", "[mkv_stream_reader][test_mkv_stream_reader.cpp]") {
    MKVStreamReader reader;
    
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    REQUIRE(reader.open(TEST_MKV_FILE));
    
    SECTION("Read first 100 packets for testing") {
        AVPacket* packet = av_packet_alloc();
        REQUIRE(packet != nullptr);
        
        int video_packets = 0;
        int audio_packets = 0;
        int total_packets = 0;
        
        while (reader.readNextPacket(packet) && total_packets < 100) {
            if (reader.isVideoPacket(packet)) {
                video_packets++;
            } else if (reader.isAudioPacket(packet)) {
                audio_packets++;
            }
            
            av_packet_unref(packet);
            total_packets++;
        }
        
        av_packet_free(&packet);
        
        // Verify we read video and audio packets
        REQUIRE(video_packets > 0);
        REQUIRE(total_packets > 0);
        
        std::cout << "Read " << total_packets << " packets" << std::endl;
        std::cout << "Video packets: " << video_packets << std::endl;
        std::cout << "Audio packets: " << audio_packets << std::endl;
    }
    
    reader.close();
}

TEST_CASE("MKVStreamReader EOF handling", "[mkv_stream_reader][test_mkv_stream_reader.cpp]") {
    MKVStreamReader reader;
    
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    REQUIRE(reader.open(TEST_MKV_FILE));
    
    // Read all packets until EOF
    AVPacket* packet = av_packet_alloc();
    REQUIRE(packet != nullptr);
    
    int total_packets = 0;
    while (reader.readNextPacket(packet)) {
        av_packet_unref(packet);
        total_packets++;
    }
    
    av_packet_free(&packet);
    
    // Verify EOF status
    REQUIRE(reader.isEOF());
    REQUIRE(total_packets > 0);
    
    std::cout << "Total packets read: " << total_packets << std::endl;
    
    reader.close();
}

TEST_CASE("MKVStreamReader reopen capability", "[mkv_stream_reader][test_mkv_stream_reader.cpp]") {
    MKVStreamReader reader;
    
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    // Test multiple open/close cycles for the same file
    for (int i = 0; i < 3; i++) {
        REQUIRE(reader.open(TEST_MKV_FILE));
        REQUIRE(reader.isOpen());
        REQUIRE_FALSE(reader.isEOF());
        
        auto info = reader.getStreamInfo();
        REQUIRE(info.video_stream_index >= 0);
        
        reader.close();
        REQUIRE_FALSE(reader.isOpen());
    }
}

TEST_CASE("MKVStreamReader multiple stream info calls", "[mkv_stream_reader][test_mkv_stream_reader.cpp]") {
    MKVStreamReader reader;
    
    std::ifstream file(TEST_MKV_FILE);
    if (!file.good()) {
        SKIP("Test file " << TEST_MKV_FILE << " not found");
    }
    
    REQUIRE(reader.open(TEST_MKV_FILE));
    
    // Multiple calls should return the same information
    auto info1 = reader.getStreamInfo();
    auto info2 = reader.getStreamInfo();
    
    REQUIRE(info1.video_stream_index == info2.video_stream_index);
    REQUIRE(info1.audio_stream_index == info2.audio_stream_index);
    REQUIRE(info1.width == info2.width);
    REQUIRE(info1.height == info2.height);
    REQUIRE(info1.duration == info2.duration);
    REQUIRE(info1.video_codec == info2.video_codec);
    
    reader.close();
}

TEST_CASE("MKVStreamReader error handling", "[mkv_stream_reader][test_mkv_stream_reader.cpp]") {
    MKVStreamReader reader;
    
    // Operations on unopened file should safely fail
    REQUIRE_FALSE(reader.isOpen());
    REQUIRE_FALSE(reader.isEOF());
    
    AVPacket* packet = av_packet_alloc();
    REQUIRE_FALSE(reader.readNextPacket(packet));
    REQUIRE_FALSE(reader.isVideoPacket(packet));
    REQUIRE_FALSE(reader.isAudioPacket(packet));
    av_packet_free(&packet);
    
    REQUIRE(reader.getVideoCodecParameters() == nullptr);
    REQUIRE(reader.getAudioCodecParameters() == nullptr);
    
    // These should not crash
    reader.close();
}

