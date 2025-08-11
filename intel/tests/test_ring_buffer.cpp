#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../src/ring_buffer.h"
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <numeric>

TEST_CASE("RingBuffer basic operations", "[test_ring_buffer.cpp]") {
    SECTION("Construction and capacity") {
        RingBuffer buffer(1024, 2);
        
        REQUIRE(buffer.channels() == 2);
        REQUIRE(buffer.capacity() >= 1024);
        REQUIRE(buffer.is_empty());
    }
    
    SECTION("Single write and read") {
        RingBuffer buffer(64, 2);
        std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f}; // 2 samples, stereo
        
        size_t written = buffer.write(test_data.data(), 2);
        REQUIRE(written == 2);
        REQUIRE(buffer.available_for_read() == 2);
        REQUIRE(buffer.available_for_write() > 0);
        
        std::vector<float> read_data(4);
        size_t read = buffer.read(read_data.data(), 2);
        REQUIRE(read == 2);
        REQUIRE(read_data[0] == 1.0f);
        REQUIRE(read_data[1] == 2.0f);
        REQUIRE(read_data[2] == 3.0f);
        REQUIRE(read_data[3] == 4.0f);
        
        REQUIRE(buffer.is_empty());
    }
    
    SECTION("Multiple write and read operations") {
        RingBuffer buffer(64, 1);
        std::vector<float> write_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        // Write in chunks
        size_t written1 = buffer.write(write_data.data(), 2);
        size_t written2 = buffer.write(write_data.data() + 2, 3);
        REQUIRE(written1 == 2);
        REQUIRE(written2 == 3);
        REQUIRE(buffer.available_for_read() == 5);
        
        // Read in chunks
        std::vector<float> read_data(5);
        size_t read1 = buffer.read(read_data.data(), 2);
        size_t read2 = buffer.read(read_data.data() + 2, 3);
        REQUIRE(read1 == 2);
        REQUIRE(read2 == 3);
        
        for (size_t i = 0; i < 5; ++i) {
            REQUIRE(read_data[i] == write_data[i]);
        }
    }
    
    SECTION("Wrap around behavior") {
        RingBuffer buffer(4, 1);
        std::vector<float> data1 = {1.0f, 2.0f};
        std::vector<float> data2 = {3.0f, 4.0f};
        std::vector<float> data3 = {5.0f, 6.0f};
        
        // Fill buffer partially
        buffer.write(data1.data(), 2);
        buffer.write(data2.data(), 2); // Should reach near capacity
        
        // Read some data to make space
        std::vector<float> read_buffer(2);
        buffer.read(read_buffer.data(), 2);
        
        // Write new data (should wrap around)
        size_t written = buffer.write(data3.data(), 2);
        REQUIRE(written == 2);
        
        // Verify the remaining data
        std::vector<float> remaining(4);
        size_t read = buffer.read(remaining.data(), 4);
        REQUIRE(read == 4);
        REQUIRE(remaining[0] == 3.0f);
        REQUIRE(remaining[1] == 4.0f);
        REQUIRE(remaining[2] == 5.0f);
        REQUIRE(remaining[3] == 6.0f);
    }
    
    SECTION("Buffer overflow protection") {
        RingBuffer buffer(4, 1);
        std::vector<float> large_data(10, 1.0f);
        
        // Try to write more than capacity
        size_t written = buffer.write(large_data.data(), 10);
        REQUIRE(written < 10); // Should be limited by available space
        REQUIRE(written > 0);   // But some data should be written
        
        // Buffer should not be corrupted
        std::vector<float> read_data(written);
        size_t read = buffer.read(read_data.data(), written);
        REQUIRE(read == written);
        
        for (size_t i = 0; i < written; ++i) {
            REQUIRE(read_data[i] == 1.0f);
        }
    }
    
    SECTION("Clear operation") {
        RingBuffer buffer(64, 2);
        std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f};
        
        buffer.write(test_data.data(), 2);
        REQUIRE_FALSE(buffer.is_empty());
        
        buffer.clear();
        REQUIRE(buffer.is_empty());
        REQUIRE(buffer.available_for_read() == 0);
        REQUIRE(buffer.available_for_write() > 0);
    }
}

TEST_CASE("RingBuffer buffer duration calculation", "[test_ring_buffer.cpp]") {
    SECTION("Duration calculation for different sample rates") {
        RingBuffer buffer(48000, 2); // 1 second buffer at 48kHz
        std::vector<float> data(2400 * 2, 1.0f); // 50ms worth of stereo data
        
        buffer.write(data.data(), 2400);
        
        double duration_48k = buffer.buffer_duration_ms(48000.0);
        REQUIRE(duration_48k == Catch::Approx(50.0).margin(1.0));
        
        double duration_44k = buffer.buffer_duration_ms(44100.0);
        REQUIRE(duration_44k == Catch::Approx(54.42).margin(1.0));
    }
}

TEST_CASE("RingBuffer concurrent access", "[test_ring_buffer.cpp]") {
    SECTION("Producer-consumer pattern") {
        RingBuffer buffer(1024, 1);
        std::atomic<bool> stop_flag{false};
        std::atomic<size_t> total_written{0};
        std::atomic<size_t> total_read{0};
        
        // Producer thread
        std::thread producer([&]() {
            std::vector<float> data(100);
            std::iota(data.begin(), data.end(), 1.0f);
            
            while (!stop_flag.load()) {
                size_t written = buffer.write(data.data(), data.size());
                total_written.fetch_add(written);
                
                if (written == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
        
        // Consumer thread
        std::thread consumer([&]() {
            std::vector<float> data(50);
            
            while (!stop_flag.load()) {
                size_t read = buffer.read(data.data(), data.size());
                total_read.fetch_add(read);
                
                if (read == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
        
        // Let threads run
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        stop_flag.store(true);
        
        producer.join();
        consumer.join();
        
        // Verify some data was processed
        REQUIRE(total_written.load() > 0);
        REQUIRE(total_read.load() > 0);
        
        // Read remaining data
        std::vector<float> remaining_data(buffer.available_for_read());
        if (!remaining_data.empty()) {
            size_t final_read = buffer.read(remaining_data.data(), remaining_data.size());
            total_read.fetch_add(final_read);
        }
        
        // Total read should not exceed total written
        REQUIRE(total_read.load() <= total_written.load());
    }
}

TEST_CASE("RingBuffer stress test", "[test_ring_buffer.cpp]") {
    SECTION("Random size operations") {
        RingBuffer buffer(512, 2);
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> size_dist(1, 50);
        std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
        
        std::vector<float> write_buffer(100);
        std::vector<float> read_buffer(100);
        
        for (int iter = 0; iter < 1000; ++iter) {
            // Random write
            size_t write_count = size_dist(rng);
            write_count = std::min(write_count, buffer.available_for_write());
            
            if (write_count > 0) {
                for (size_t i = 0; i < write_count * 2; ++i) {
                    write_buffer[i] = value_dist(rng);
                }
                
                size_t written = buffer.write(write_buffer.data(), write_count);
                REQUIRE(written <= write_count);
            }
            
            // Random read
            size_t read_count = size_dist(rng);
            read_count = std::min(read_count, buffer.available_for_read());
            
            if (read_count > 0) {
                size_t read = buffer.read(read_buffer.data(), read_count);
                REQUIRE(read <= read_count);
            }
            
            // Verify buffer invariants (no reserved slot needed with our implementation)
            REQUIRE(buffer.available_for_read() + buffer.available_for_write() == buffer.capacity());
        }
    }
}