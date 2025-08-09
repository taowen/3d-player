---
alwaysApply: true
applyTo: '**'
---

# 3D Player

## 开发流程

遵循测试驱动、小步迭代的原则

### 1. 调研阶段
- 主要是依赖 ffmpeg。可以使用 deepwiki 查不熟悉的接口
- 先调研清楚当前任务的外部依赖，以及所有涉及的上下游文件
- 确认测试存在和测试逻辑的合理性

### 2. 接口设计阶段
- **从最简单的公开接口开始设计**，不要上来就把所有可能的功能都加上
- **不要上来就定义 private method**，实现时再按需添加
- 专注于核心功能，避免过度设计
- 通过多轮简化，去掉所有不必要的复杂性

### 3. 测试驱动开发
- **⚠️ 先写测试，再写实现**：绝不允许在没有测试的情况下编写实现代码
- **一次只写一个测试 section**：每个 TEST_CASE 代表一个完整的功能点，完成一个再写下一个
- **最小化实现**：写最简单的实现让当前测试通过，不要提前考虑未来的需求
- **逐步迭代**：每次只添加一个小功能，保持测试始终通过
- **明确期望**：测试要清晰地描述期望的行为，作为实现的规范

### 4. 小步迭代原则
- 每次只做一件事，做完一件事再做下一件事
- 不要在一件事还没有做完之前就开始添加性能监控等次要的新功能
- 及时把代码的改动同步到 README.md 方便新人快速上手

### 5. 代码简化原则
- 如果一个功能或接口被质疑是否需要，优先选择删除
- 保持接口最小化，专注于核心功能
- 避免添加"可能有用"的功能，只添加"确实需要"的功能

## 文件结构
```
src/
├── main.cpp                  # 3D Player主程序入口
├── mkv_stream_reader.h       # MKV流读取器头文件
├── mkv_stream_reader.cpp     # MKV流读取器实现
├── hw_video_decoder.h        # DirectX 11硬件视频解码器头文件
├── hw_video_decoder.cpp      # DirectX 11硬件视频解码器实现
├── audio_decoder.h           # 音频解码器头文件
├── audio_decoder.cpp         # 音频解码器实现
├── rgb_video_decoder.h       # RGB视频解码器头文件
├── rgb_video_decoder.cpp     # RGB视频解码器实现
├── audio_player.h            # 音频播放器头文件
├── audio_player.cpp          # 音频播放器实现
├── video_player.h            # 视频播放器头文件
├── video_player.cpp          # 视频播放器实现
├── audio_video_player.h      # 音视频播放器头文件
├── audio_video_player.cpp    # 音视频播放器实现
├── audio_only_player.cpp     # 纯音频播放器程序入口
└── fullscreen_quad.hlsl      # 全屏四边形着色器
tests/
├── test_mkv_stream_reader.cpp # MKV流读取器测试
├── test_hw_video_decoder.cpp  # 硬件视频解码器测试
├── test_audio_decoder.cpp    # 音频解码器测试
├── test_rgb_video_decoder.cpp # RGB视频解码器测试
├── test_audio_player.cpp     # 音频播放器测试
├── test_video_player.cpp     # 视频播放器测试
└── test_audio_video_player.cpp # 音视频播放器测试
test_data/
├── sample_hw.mkv             # 测试文件 - H.264编码，支持硬件解码
└── sample_with_audio.mkv     # 测试文件 - H.264编码，支持硬件解码，包含AAC音频
CMakeLists.txt               # CMake构建配置文件
3d-player.ps1               # 3D Player构建运行脚本
audio-only.ps1              # 纯音频播放器构建运行脚本
test.ps1                    # 单个测试运行脚本
test-all.ps1               # 所有测试运行脚本
```

## 核心组件

### MKVStreamReader
MKV文件流读取器，提供 Pull-style 接口，支持视频和音频流的解析和包读取。详细接口请参考 `src/mkv_stream_reader.h`。

### HwVideoDecoder  
DirectX 11 硬件视频解码器，使用 FFmpeg AVBufferPool 进行内存管理。详细接口请参考 `src/hw_video_decoder.h`。

### AudioDecoder
音频解码器，使用 FFmpeg 进行音频解码。支持常见的音频编解码器（AAC、MP3、Opus等），输出格式为 PCM 音频帧。详细接口请参考 `src/audio_decoder.h`。

### RgbVideoDecoder
RGB 视频解码器，基于 HwVideoDecoder 并提供 RGB 转换功能。内部构造 HwVideoDecoder，使用 VideoProcessorBlt 将 YUV 格式转换为 RGB D3D11 纹理。详细接口请参考 `src/rgb_video_decoder.h`。

### AudioPlayer
音频播放器，支持时间驱动的音频播放控制。基于 AudioDecoder 提供音频解码和 WASAPI 音频输出，支持 onTimer 接口进行时间驱动的音频播放。
详细接口请参考 `src/audio_player.h`。

### VideoPlayer
视频播放器，支持时间驱动的播放控制和预解码缓冲。基于 RgbVideoDecoder 提供 RGB 纹理输出，支持 onTimer 接口进行时间驱动的帧切换。专注于纯视频播放，不处理音频。
详细接口请参考 `src/video_player.h`。

### AudioVideoPlayer
音视频播放器，组合 AudioPlayer 和 VideoPlayer 功能。提供统一的接口管理音频和视频播放，支持时间驱动的音视频同步播放。
详细接口请参考 `src/audio_video_player.h`。


### 3D Player 应用程序
基于 AudioVideoPlayer 组件构建的完整音视频播放应用程序，提供 Windows 窗口播放体验。

**核心特性**：
- 完整的 Win32 窗口应用程序
- 基于 AudioVideoPlayer 的交换链渲染和音频输出
- 时间驱动的自动音视频同步播放
- 支持命令行参数传入音视频文件路径
- 自适应视频分辨率窗口，自动居中显示
- 简洁的用户交互：ESC 键退出

**实现细节**：
- 使用 HwVideoDecoder 创建的 D3D11 设备确保设备一致性
- 创建 DXGI 交换链绑定到窗口
- 消息循环中调用 `AudioVideoPlayer::onTimer` 进行时间驱动的播放
- 自动检测音视频播放结束并退出程序

详细实现请参考 `src/main.cpp`。


## 核心决策

### 设计原则：单一实现方案
- **⚠️ 严禁回退机制**：代码中不允许添加任何回退逻辑或备用方案
- **单一代码路径**：每个功能只有一种实现方式，确保代码简洁可维护
- **快速失败**：遇到错误直接失败并报告，不尝试备用方案
- **明确依赖**：所有依赖都是硬性要求，不支持降级使用

### DirectX 11 硬件加速
- **⚠️ 强制硬件解码**：本软件严格要求硬件解码，绝不允许软件解码回退
- **支持编解码器**：H.264 (h264_d3d11va)、HEVC (hevc_d3d11va)、AV1 (av1_d3d11va)
- **输出格式**：仅支持硬件像素格式（AV_PIX_FMT_D3D11、AV_PIX_FMT_D3D11VA_VLD、AV_PIX_FMT_CUDA、AV_PIX_FMT_VULKAN）
- **RGB 转换**：使用 VideoProcessorBlt 将 YUV 转换为 RGB D3D11 纹理
- **格式协商**：自动选择最佳的硬件像素格式，无硬件格式时直接失败

### FFmpeg AVBufferPool 缓冲管理
- **自动内存管理**：使用 FFmpeg 内置的 AVBufferPool 系统
- **线程安全**：内置原子操作，支持多线程环境
- **高效重用**：自动回收和重用缓冲区，减少GPU内存分配开销
- **动态调整**：根据解码器需求自动分配适量的参考帧缓冲区

## 构建测试

**当前环境**：Git Bash + Windows (AMD64)

### 编译构建
```bash
# 构建并运行 3D Player（支持可选视频文件参数）
powershell -ExecutionPolicy Bypass -File 3d-player.ps1 -VideoFile test_data/sample_hw.mkv
# 仅构建不运行
powershell -ExecutionPolicy Bypass -File 3d-player.ps1 -BuildOnly

# 构建并运行单个测试 `build/Debug/integration-test.exe`
powershell -ExecutionPolicy Bypass -File test.ps1 "[测试文件名.cpp]"
# 例如 test_mkv_stream_reader.cpp 修改了，要跑这个测试
# [test_mkv_stream_reader.cpp] 是写在了 test_mkv_stream_reader.cpp 文件中的 catch2 测试tag
# 每个新建的测试文件也需要遵循这个惯例，把自己的文件名做为 catch2 测试的 tag
powershell -ExecutionPolicy Bypass -File test.ps1 "[test_mkv_stream_reader.cpp]"

# 构建并运行所有测试 `build/Debug/integration-test.exe`
powershell -ExecutionPolicy Bypass -File test-all.ps1

```

- **测试框架**：c++ catch2 测试框架
- **测试运行目录**：测试程序在项目根目录 (`3d-player\`) 运行
- **文件路径**：测试中所有相对路径都是相对于项目根目录的
- **测试文件**：`test_data\sample_hw.mkv` 从项目根目录访问

### 3D Player 使用方法
- **功能**：从头到尾顺序播放音视频到 Windows 窗口
- **控制**：按 ESC 键退出，音视频播放完成后自动退出
- **窗口**：自适应视频分辨率，自动居中显示
- **要求**：必须支持 DirectX 11 硬件解码
- **音频**：自动检测音频流，支持 WASAPI 音频输出

⚠️ **注意**：
- Git Bash 环境下使用正斜杠路径：`test_data/sample_hw.mkv`
- 可执行文件路径：`build/Debug/3d_player.exe`

---

## 🔧 维护指南

- **非必要不更新**：只有当修改影响到新人快速上手时才更新文档
- 保持简洁，只记录与快速上手改代码相关的必要核心信息
- 避免添加实现细节、性能优化描述等非关键信息
- 文档和代码不同步的时候及时更新 README.md，避免文档过时

### 需要更新的内容：
1. **文件结构** - 添加/删除文件时更新
2. **核心API** - 修改公共接口时更新
3. **构建测试** - 改变构建方式时更新

### 不能更新的内容

不要更新"维护指南"这个章节，保持这个章节是文档最后一个部分。
不能新增新的大章节。