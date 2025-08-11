#pragma once

extern "C" {
#include <libavutil/frame.h>
#include <libswresample/swresample.h>
}

class AudioFormatConverter {
public:
    AudioFormatConverter();
    ~AudioFormatConverter();

    bool configure(int src_sample_rate, AVSampleFormat src_format, int src_channels,
                  int dst_sample_rate = 48000, AVSampleFormat dst_format = AV_SAMPLE_FMT_FLT, 
                  int dst_channels = 2);
    
    int convert(AVFrame* input_frame, float** output_samples, int* output_sample_count);
    
    int output_sample_rate() const { return dst_sample_rate_; }
    int output_channels() const { return dst_channels_; }

private:
    SwrContext* swr_ctx_;
    int src_sample_rate_;
    AVSampleFormat src_format_;
    int src_channels_;
    int dst_sample_rate_;
    AVSampleFormat dst_format_;
    int dst_channels_;
    
    uint8_t** converted_data_;
    int converted_linesize_;
    int max_converted_samples_;
    
    void cleanup();
    bool allocate_output_buffer(int sample_count);
};