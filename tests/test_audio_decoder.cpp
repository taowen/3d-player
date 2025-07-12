#include <catch2/catch_test_macros.hpp>
#include "../src/audio_decoder.h"
#include <iostream>

extern "C" {
#include <libavutil/samplefmt.h>
}


TEST_CASE("AudioDecoder basic functionality", "[audio_decoder]") {
    AudioDecoder decoder;
    
    // 打开视频文件（注意：当前测试文件只包含视频流，没有音频流）
    bool opened = decoder.open("test_data/sample_hw.mkv");
    REQUIRE_FALSE(opened);  // 应该失败，因为没有音频流
    REQUIRE_FALSE(decoder.isOpen());
    
    // 测试错误处理
    AudioDecoder::DecodedFrame frame;
    bool result = decoder.readNextFrame(frame);
    REQUIRE_FALSE(result);  // 应该失败，因为解码器未打开
    REQUIRE_FALSE(frame.is_valid);
    
    // 检查未打开状态
    REQUIRE_FALSE(decoder.isOpen());
    REQUIRE_FALSE(decoder.isEOF());
    
    // 获取流读取器应该返回有效指针
    REQUIRE(decoder.getStreamReader() != nullptr);
    
    // 关闭应该安全
    decoder.close();
    REQUIRE_FALSE(decoder.isOpen());
} 