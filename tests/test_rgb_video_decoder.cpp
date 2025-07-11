#include "../src/rgb_video_decoder.h"
#include <iostream>
#include <string>
#include <cassert>

void test_rgb_video_decoder_basic() {
    std::cout << "Testing RgbVideoDecoder basic functionality..." << std::endl;
    
    RgbVideoDecoder decoder;
    
    // 测试文件不存在的情况
    assert(!decoder.open("nonexistent_file.mkv"));
    assert(!decoder.isOpen());
    
    // 测试打开测试文件
    std::string test_file = "test_data/sample_hw.mkv";
    bool opened = decoder.open(test_file);
    
    if (!opened) {
        std::cerr << "Warning: Could not open test file: " << test_file << std::endl;
        std::cerr << "This may be expected if test file doesn't exist or hardware decoder is not available." << std::endl;
        return;
    }
    
    assert(decoder.isOpen());
    assert(!decoder.isEOF());
    
    std::cout << "Video dimensions: " << decoder.getWidth() << "x" << decoder.getHeight() << std::endl;
    
    // 测试读取帧
    RgbVideoDecoder::DecodedRgbFrame frame;
    int frame_count = 0;
    int max_frames = 10; // 限制测试帧数
    
    while (decoder.readNextFrame(frame) && frame_count < max_frames) {
        assert(frame.is_valid);
        assert(frame.hw_frame.frame != nullptr);
        assert(frame.rgb_texture != nullptr);
        
        std::cout << "Successfully decoded frame " << frame_count + 1 << std::endl;
        
        // 释放 FFmpeg 帧内存
        av_frame_free(&frame.hw_frame.frame);
        
        frame_count++;
    }
    
    std::cout << "Total frames processed: " << frame_count << std::endl;
    
    // 关闭解码器
    decoder.close();
    assert(!decoder.isOpen());
    
    std::cout << "RgbVideoDecoder basic functionality test passed!" << std::endl;
}

void test_rgb_video_decoder_hw_access() {
    std::cout << "Testing RgbVideoDecoder hardware decoder access..." << std::endl;
    
    RgbVideoDecoder decoder;
    
    // 测试获取硬件解码器
    HwVideoDecoder* hw_decoder = decoder.getHwDecoder();
    assert(hw_decoder != nullptr);
    
    std::string test_file = "test_data/sample_hw.mkv";
    if (!decoder.open(test_file)) {
        std::cerr << "Warning: Could not open test file for hardware access test" << std::endl;
        return;
    }
    
    // 测试硬件解码器访问
    assert(hw_decoder->isOpen());
    assert(hw_decoder->getStreamReader() != nullptr);
    
    decoder.close();
    
    std::cout << "RgbVideoDecoder hardware decoder access test passed!" << std::endl;
}

void test_rgb_video_decoder_error_handling() {
    std::cout << "Testing RgbVideoDecoder error handling..." << std::endl;
    
    RgbVideoDecoder decoder;
    
    // 测试未打开时的操作
    RgbVideoDecoder::DecodedRgbFrame frame;
    assert(!decoder.readNextFrame(frame));
    assert(!decoder.isOpen());
    assert(!decoder.isEOF());
    assert(decoder.getWidth() == 0);
    assert(decoder.getHeight() == 0);
    
    // 测试多次关闭
    decoder.close();
    decoder.close(); // 应该安全
    
    std::cout << "RgbVideoDecoder error handling test passed!" << std::endl;
}

int main() {
    try {
        test_rgb_video_decoder_basic();
        test_rgb_video_decoder_hw_access();
        test_rgb_video_decoder_error_handling();
        
        std::cout << "All RgbVideoDecoder tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
} 