#pragma once

#include <atomic>
#include <memory>
#include <cstddef>

class RingBuffer {
public:
    explicit RingBuffer(size_t sample_count, size_t channels = 2);
    ~RingBuffer();

    size_t write(const float* samples, size_t sample_count);
    size_t read(float* samples, size_t sample_count);
    
    size_t available_for_write() const;
    size_t available_for_read() const;
    
    size_t capacity() const { return capacity_; }
    size_t channels() const { return channels_; }
    
    void clear();
    
    bool is_empty() const;
    
    double buffer_duration_ms(double sample_rate) const;

private:
    const size_t capacity_;
    const size_t channels_;
    const size_t buffer_size_bytes_;
    
    std::unique_ptr<float[]> buffer_;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    
    size_t mask_;
    
    size_t next_power_of_two(size_t n);
};