#include "ring_buffer.h"
#include <algorithm>
#include <cstring>

RingBuffer::RingBuffer(size_t sample_count, size_t channels)
    : capacity_(next_power_of_two(sample_count))
    , channels_(channels)
    , buffer_size_bytes_(capacity_ * channels_ * sizeof(float))
    , buffer_(std::make_unique<float[]>(capacity_ * channels_))
    , mask_(capacity_ - 1) {
}

RingBuffer::~RingBuffer() = default;

size_t RingBuffer::write(const float* samples, size_t sample_count) {
    const size_t available = available_for_write();
    const size_t to_write = std::min(sample_count, available);
    
    if (to_write == 0) {
        return 0;
    }
    
    const size_t write_idx = write_pos_.load(std::memory_order_acquire);
    const size_t samples_per_frame = channels_;
    
    for (size_t i = 0; i < to_write; ++i) {
        const size_t pos = (write_idx + i) & mask_;
        for (size_t ch = 0; ch < samples_per_frame; ++ch) {
            buffer_[pos * samples_per_frame + ch] = samples[i * samples_per_frame + ch];
        }
    }
    
    write_pos_.store(write_idx + to_write, std::memory_order_release);
    return to_write;
}

size_t RingBuffer::read(float* samples, size_t sample_count) {
    const size_t available = available_for_read();
    const size_t to_read = std::min(sample_count, available);
    
    if (to_read == 0) {
        return 0;
    }
    
    const size_t read_idx = read_pos_.load(std::memory_order_acquire);
    const size_t samples_per_frame = channels_;
    
    for (size_t i = 0; i < to_read; ++i) {
        const size_t pos = (read_idx + i) & mask_;
        for (size_t ch = 0; ch < samples_per_frame; ++ch) {
            samples[i * samples_per_frame + ch] = buffer_[pos * samples_per_frame + ch];
        }
    }
    
    read_pos_.store(read_idx + to_read, std::memory_order_release);
    return to_read;
}

size_t RingBuffer::available_for_write() const {
    const size_t write_idx = write_pos_.load(std::memory_order_acquire);
    const size_t read_idx = read_pos_.load(std::memory_order_acquire);
    
    const size_t used = write_idx - read_idx;
    // Allow full utilization of capacity by using a different method to detect full/empty
    return capacity_ - used;
}

size_t RingBuffer::available_for_read() const {
    const size_t write_idx = write_pos_.load(std::memory_order_acquire);
    const size_t read_idx = read_pos_.load(std::memory_order_acquire);
    
    return write_idx - read_idx;
}

void RingBuffer::clear() {
    read_pos_.store(0, std::memory_order_release);
    write_pos_.store(0, std::memory_order_release);
}

bool RingBuffer::is_empty() const {
    return available_for_read() == 0;
}

double RingBuffer::buffer_duration_ms(double sample_rate) const {
    const size_t available_samples = available_for_read();
    return (available_samples * 1000.0) / sample_rate;
}

size_t RingBuffer::next_power_of_two(size_t n) {
    if (n == 0) return 1;
    
    size_t result = 1;
    while (result < n) {
        result <<= 1;
    }
    return result;
}