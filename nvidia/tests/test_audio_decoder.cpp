#include <catch2/catch_test_macros.hpp>
#include "../src/audio_decoder.h"
#include <iostream>

extern "C" {
#include <libavutil/samplefmt.h>
}

TEST_CASE("AudioDecoder with audio stream", "[audio_decoder][test_audio_decoder.cpp]") {
    AudioDecoder decoder;
    
    // 打开包含音频流的文件
    bool opened = decoder.open("test_data/sample_with_audio.mkv");
    REQUIRE(opened);  // 应该成功，因为有音频流
    REQUIRE(decoder.isOpen());
    
    // 测试流读取器
    MKVStreamReader* reader = decoder.getStreamReader();
    REQUIRE(reader != nullptr);
    REQUIRE(reader->isOpen());
    
    // 读取音频帧
    AudioDecoder::DecodedFrame frame;
    bool result = decoder.readNextFrame(frame);
    REQUIRE(result);  // 应该成功读取到音频帧
    REQUIRE(frame.is_valid);
    REQUIRE(frame.frame != nullptr);
    
    // 检查音频帧属性
    REQUIRE(frame.frame->format == AV_SAMPLE_FMT_FLTP);  // AAC 解码输出格式
    REQUIRE(frame.frame->sample_rate == 44100);          // 采样率
    REQUIRE(frame.frame->ch_layout.nb_channels == 1);    // 单声道
    REQUIRE(frame.frame->nb_samples > 0);                // 有采样数据
    
    // 释放当前帧
    av_frame_free(&frame.frame);
    
    // 读取更多帧
    int frame_count = 1;
    while (decoder.readNextFrame(frame) && frame_count < 10) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        av_frame_free(&frame.frame);
        frame_count++;
    }
    
    // 应该读取到多个帧
    REQUIRE(frame_count > 1);
    
    // 关闭解码器
    decoder.close();
    REQUIRE_FALSE(decoder.isOpen());
}


TEST_CASE("AudioDecoder EOF detection", "[audio_decoder][test_audio_decoder.cpp]") {
    AudioDecoder decoder;
    
    // 打开包含音频流的文件
    bool opened = decoder.open("test_data/sample_with_audio.mkv");
    REQUIRE(opened);
    
    // 读取所有帧直到 EOF
    AudioDecoder::DecodedFrame frame;
    int frame_count = 0;
    while (decoder.readNextFrame(frame)) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        av_frame_free(&frame.frame);
        frame_count++;
    }
    
    // 应该读取到一些帧
    REQUIRE(frame_count > 0);
    
    // 应该达到 EOF
    REQUIRE(decoder.isEOF());
    
    // 尝试再次读取，应该失败
    bool result = decoder.readNextFrame(frame);
    REQUIRE_FALSE(result);
    REQUIRE_FALSE(frame.is_valid);
    
    decoder.close();
}


TEST_CASE("AudioDecoder resource management", "[audio_decoder][test_audio_decoder.cpp]") {
    AudioDecoder decoder;
    
    // 测试多次打开和关闭
    for (int i = 0; i < 3; i++) {
        bool opened = decoder.open("test_data/sample_with_audio.mkv");
        REQUIRE(opened);
        REQUIRE(decoder.isOpen());
        
        // 读取至少一帧
        AudioDecoder::DecodedFrame frame;
        bool result = decoder.readNextFrame(frame);
        REQUIRE(result);
        REQUIRE(frame.is_valid);
        av_frame_free(&frame.frame);
        
        // 关闭
        decoder.close();
        REQUIRE_FALSE(decoder.isOpen());
    }
    
    // 测试重新打开覆盖
    bool opened1 = decoder.open("test_data/sample_with_audio.mkv");
    REQUIRE(opened1);
    REQUIRE(decoder.isOpen());
    
    // 不关闭直接打开另一个文件
    bool opened2 = decoder.open("test_data/sample_with_audio.mkv");
    REQUIRE(opened2);
    REQUIRE(decoder.isOpen());
    
    decoder.close();
    REQUIRE_FALSE(decoder.isOpen());
}


TEST_CASE("AudioDecoder error handling", "[audio_decoder][test_audio_decoder.cpp]") {
    AudioDecoder decoder;
    
    // 测试打开不存在的文件
    bool opened = decoder.open("non_existent_file.mkv");
    REQUIRE_FALSE(opened);
    REQUIRE_FALSE(decoder.isOpen());
    
    // 测试在未打开状态下操作
    AudioDecoder::DecodedFrame frame;
    bool result = decoder.readNextFrame(frame);
    REQUIRE_FALSE(result);
    REQUIRE_FALSE(frame.is_valid);
    
    // 测试 EOF 状态
    REQUIRE_FALSE(decoder.isEOF());
    
    // 测试关闭未打开的解码器
    decoder.close();  // 应该安全
    REQUIRE_FALSE(decoder.isOpen());
    
    // 测试流读取器
    MKVStreamReader* reader = decoder.getStreamReader();
    REQUIRE(reader != nullptr);
    REQUIRE_FALSE(reader->isOpen());
} 