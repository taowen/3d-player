#include "audio_format_converter.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

AudioFormatConverter::AudioFormatConverter()
    : swr_ctx_(nullptr)
    , src_sample_rate_(0)
    , src_format_(AV_SAMPLE_FMT_NONE)
    , src_channels_(0)
    , dst_sample_rate_(0)
    , dst_format_(AV_SAMPLE_FMT_NONE)
    , dst_channels_(0)
    , converted_data_(nullptr)
    , converted_linesize_(0)
    , max_converted_samples_(0) {
}

AudioFormatConverter::~AudioFormatConverter() {
    cleanup();
}

bool AudioFormatConverter::configure(int src_sample_rate, AVSampleFormat src_format, int src_channels,
                                   int dst_sample_rate, AVSampleFormat dst_format, int dst_channels) {
    cleanup();
    
    src_sample_rate_ = src_sample_rate;
    src_format_ = src_format;
    src_channels_ = src_channels;
    dst_sample_rate_ = dst_sample_rate;
    dst_format_ = dst_format;
    dst_channels_ = dst_channels;
    
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) {
        return false;
    }
    
    AVChannelLayout src_ch_layout, dst_ch_layout;
    av_channel_layout_default(&src_ch_layout, src_channels);
    av_channel_layout_default(&dst_ch_layout, dst_channels);
    
    av_opt_set_chlayout(swr_ctx_, "in_chlayout", &src_ch_layout, 0);
    av_opt_set_int(swr_ctx_, "in_sample_rate", src_sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx_, "in_sample_fmt", src_format, 0);
    
    av_opt_set_chlayout(swr_ctx_, "out_chlayout", &dst_ch_layout, 0);
    av_opt_set_int(swr_ctx_, "out_sample_rate", dst_sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx_, "out_sample_fmt", dst_format, 0);
    
    av_channel_layout_uninit(&src_ch_layout);
    av_channel_layout_uninit(&dst_ch_layout);
    
    if (swr_init(swr_ctx_) < 0) {
        swr_free(&swr_ctx_);
        return false;
    }
    
    return true;
}

int AudioFormatConverter::convert(AVFrame* input_frame, float** output_samples, int* output_sample_count) {
    if (!swr_ctx_ || !input_frame || !output_samples || !output_sample_count) {
        return -1;
    }
    
    int input_samples = input_frame->nb_samples;
    int estimated_output_samples = swr_get_out_samples(swr_ctx_, input_samples);
    
    if (!allocate_output_buffer(estimated_output_samples)) {
        return -1;
    }
    
    int converted_samples = swr_convert(swr_ctx_, 
                                      converted_data_, estimated_output_samples,
                                      (const uint8_t**)input_frame->data, input_samples);
    
    if (converted_samples < 0) {
        return -1;
    }
    
    *output_samples = (float*)converted_data_[0];
    *output_sample_count = converted_samples;
    
    return 0;
}

void AudioFormatConverter::cleanup() {
    if (swr_ctx_) {
        swr_free(&swr_ctx_);
    }
    
    if (converted_data_) {
        av_freep(&converted_data_[0]);
        av_freep(&converted_data_);
        converted_linesize_ = 0;
        max_converted_samples_ = 0;
    }
}

bool AudioFormatConverter::allocate_output_buffer(int sample_count) {
    if (sample_count <= max_converted_samples_ && converted_data_) {
        return true;
    }
    
    if (converted_data_) {
        av_freep(&converted_data_[0]);
        av_freep(&converted_data_);
    }
    
    int ret = av_samples_alloc_array_and_samples(&converted_data_, &converted_linesize_,
                                               dst_channels_, sample_count, dst_format_, 0);
    
    if (ret < 0) {
        max_converted_samples_ = 0;
        return false;
    }
    
    max_converted_samples_ = sample_count;
    return true;
}