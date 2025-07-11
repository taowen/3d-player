#pragma once

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
}
#include <string>

/**
 * @class MKVStreamReader
 * @brief MKV文件流读取器，支持视频和音频流的解析和包读取
 * 
 * 提供 Pull-style 接口，支持按需读取数据包而不是回调方式
 */
class MKVStreamReader {
public:
    struct StreamInfo {
        int video_stream_index = -1;
        int audio_stream_index = -1;
        std::string video_codec;
        std::string audio_codec;
        int width = 0;
        int height = 0;
        double duration = 0.0;
        AVRational video_time_base = {0, 1};
        AVRational audio_time_base = {0, 1};
        double fps = 0.0;
        int64_t video_bitrate = 0;
        int64_t audio_bitrate = 0;
        int audio_sample_rate = 0;
        int audio_channels = 0;
    };

    MKVStreamReader();
    ~MKVStreamReader();

    /**
     * @brief 打开MKV文件并解析流信息
     * @param filepath 文件路径
     * @return true 成功打开文件，false 打开失败
     */
    bool open(const std::string& filepath);
    
    /**
     * @brief 获取流信息
     * @return StreamInfo 结构体，包含视频和音频流的详细信息
     */
    StreamInfo getStreamInfo() const;
    
    /**
     * @brief 读取下一个数据包 (Pull-style)
     * @param packet 用于存储读取到的数据包的指针
     * @return true 成功读取数据包，false 读取失败或到达文件末尾
     */
    bool readNextPacket(AVPacket* packet);
    
    /**
     * @brief 检查数据包是否为视频数据包
     * @param packet 要检查的数据包
     * @return true 是视频数据包，false 不是视频数据包
     */
    bool isVideoPacket(const AVPacket* packet) const;
    
    /**
     * @brief 检查数据包是否为音频数据包
     * @param packet 要检查的数据包
     * @return true 是音频数据包，false 不是音频数据包
     */
    bool isAudioPacket(const AVPacket* packet) const;
    
    /**
     * @brief 检查文件是否已打开
     * @return true 文件已打开，false 文件未打开
     */
    bool isOpen() const;
    
    /**
     * @brief 检查是否到达文件末尾
     * @return true 到达文件末尾，false 未到达文件末尾
     */
    bool isEOF() const;
    
    /**
     * @brief 获取视频编解码器参数 (用于解码器初始化)
     * @return AVCodecParameters* 视频编解码器参数指针，未找到视频流时返回 nullptr
     */
    AVCodecParameters* getVideoCodecParameters() const;
    
    /**
     * @brief 获取音频编解码器参数 (用于解码器初始化)
     * @return AVCodecParameters* 音频编解码器参数指针，未找到音频流时返回 nullptr
     */
    AVCodecParameters* getAudioCodecParameters() const;
    
    /**
     * @brief 关闭文件并释放资源
     */
    void close();

private:
    AVFormatContext* format_context_;
    StreamInfo stream_info_;
    bool is_open_;
    bool is_eof_;
    
    bool analyzeStreams();
    void extractStreamInfo();
};