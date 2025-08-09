#include "rgb_video_decoder.h"
#include <iostream>

extern "C" {
#include <libavutil/hwcontext_d3d11va.h>
}

RgbVideoDecoder::RgbVideoDecoder()
    : hw_decoder_(std::make_unique<HwVideoDecoder>())
    , video_width_(0)
    , video_height_(0) {
}

RgbVideoDecoder::~RgbVideoDecoder() {
    close();
}

bool RgbVideoDecoder::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    // 打开硬件解码器
    if (!hw_decoder_->open(filepath)) {
        std::cerr << "Failed to open hardware decoder for: " << filepath << std::endl;
        return false;
    }
    
    // 初始化 Video Processor
    if (!initializeVideoProcessor()) {
        std::cerr << "Failed to initialize video processor" << std::endl;
        close();
        return false;
    }
    
    return true;
}

bool RgbVideoDecoder::readNextFrame(DecodedRgbFrame& frame) {
    if (!isOpen()) {
        return false;
    }
    
    frame.is_valid = false;
    
    // 读取硬件解码帧
    if (!hw_decoder_->readNextFrame(frame.hw_frame)) {
        return false;
    }
    
    // 转换为 RGB
    if (!convertYuvToRgb(frame.hw_frame.frame, frame.rgb_texture)) {
        std::cerr << "Failed to convert YUV to RGB" << std::endl;
        return false;
    }
    
    frame.is_valid = true;
    return true;
}

bool RgbVideoDecoder::isOpen() const {
    return hw_decoder_ && hw_decoder_->isOpen() && video_processor_;
}

bool RgbVideoDecoder::isEOF() const {
    return hw_decoder_ && hw_decoder_->isEOF();
}

void RgbVideoDecoder::close() {
    cleanup();
    
    if (hw_decoder_) {
        hw_decoder_->close();
    }
}

HwVideoDecoder* RgbVideoDecoder::getHwDecoder() const {
    return hw_decoder_.get();
}

int RgbVideoDecoder::getWidth() const {
    return video_width_;
}

int RgbVideoDecoder::getHeight() const {
    return video_height_;
}

bool RgbVideoDecoder::initializeVideoProcessor() {
    if (!hw_decoder_ || !hw_decoder_->isOpen()) {
        return false;
    }
    
    // 获取 D3D11 设备
    ID3D11Device* d3d11_device = hw_decoder_->getD3D11Device();
    ID3D11DeviceContext* d3d11_context = hw_decoder_->getD3D11DeviceContext();
    
    if (!d3d11_device || !d3d11_context) {
        std::cerr << "Failed to get D3D11 device or context" << std::endl;
        return false;
    }
    
    // 获取 Video Device
    HRESULT hr = d3d11_device->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&video_device_);
    if (FAILED(hr)) {
        std::cerr << "Failed to query video device" << std::endl;
        return false;
    }
    
    // 获取 Video Context
    hr = d3d11_context->QueryInterface(__uuidof(ID3D11VideoContext), (void**)&video_context_);
    if (FAILED(hr)) {
        std::cerr << "Failed to query video context" << std::endl;
        return false;
    }
    
    // 获取视频尺寸
    MKVStreamReader* stream_reader = hw_decoder_->getStreamReader();
    if (!stream_reader) {
        std::cerr << "Failed to get stream reader" << std::endl;
        return false;
    }
    
    AVCodecParameters* codecpar = stream_reader->getVideoCodecParameters();
    if (!codecpar) {
        std::cerr << "Failed to get video codec parameters" << std::endl;
        return false;
    }
    
    video_width_ = codecpar->width;
    video_height_ = codecpar->height;
    
    // 创建 Video Processor Enumerator
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC content_desc = {};
    content_desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    content_desc.InputWidth = video_width_;
    content_desc.InputHeight = video_height_;
    content_desc.OutputWidth = video_width_;
    content_desc.OutputHeight = video_height_;
    content_desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
    
    hr = video_device_->CreateVideoProcessorEnumerator(&content_desc, &video_processor_enumerator_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create video processor enumerator" << std::endl;
        return false;
    }
    
    // 创建 Video Processor
    hr = video_device_->CreateVideoProcessor(video_processor_enumerator_.Get(), 0, &video_processor_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create video processor" << std::endl;
        return false;
    }
    
    // 创建 RGB 输出纹理
    D3D11_TEXTURE2D_DESC texture_desc = {};
    texture_desc.Width = video_width_;
    texture_desc.Height = video_height_;
    texture_desc.MipLevels = 1;
    texture_desc.ArraySize = 1;
    texture_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.SampleDesc.Quality = 0;
    texture_desc.Usage = D3D11_USAGE_DEFAULT;
    texture_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    texture_desc.CPUAccessFlags = 0;
    texture_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    hr = d3d11_device->CreateTexture2D(&texture_desc, nullptr, &rgb_texture_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create RGB texture" << std::endl;
        return false;
    }
    
    // 创建 Video Processor Output View
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_desc = {};
    output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
    output_view_desc.Texture2D.MipSlice = 0;
    
    hr = video_device_->CreateVideoProcessorOutputView(rgb_texture_.Get(), video_processor_enumerator_.Get(), &output_view_desc, &output_view_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create video processor output view" << std::endl;
        return false;
    }
    
    return true;
}

bool RgbVideoDecoder::convertYuvToRgb(AVFrame* yuv_frame, ComPtr<ID3D11Texture2D>& rgb_texture) {
    if (!yuv_frame || !video_processor_ || !video_context_) {
        return false;
    }
    
    // 获取 YUV 纹理
    ID3D11Texture2D* yuv_texture = (ID3D11Texture2D*)yuv_frame->data[0];
    if (!yuv_texture) {
        std::cerr << "Failed to get YUV texture from frame" << std::endl;
        return false;
    }
    
    // 获取数组切片索引
    UINT array_slice = (UINT)(intptr_t)yuv_frame->data[1];
    
    // 获取或创建 Video Processor Input View（使用缓存）
    ComPtr<ID3D11VideoProcessorInputView> input_view = getOrCreateInputView(yuv_texture, array_slice);
    if (!input_view) {
        std::cerr << "Failed to get or create video processor input view" << std::endl;
        return false;
    }
    
    // 设置 Video Processor Stream
    D3D11_VIDEO_PROCESSOR_STREAM stream_data = {};
    stream_data.Enable = TRUE;
    stream_data.OutputIndex = 0;
    stream_data.InputFrameOrField = 0;
    stream_data.PastFrames = 0;
    stream_data.FutureFrames = 0;
    stream_data.ppPastSurfaces = nullptr;
    stream_data.ppFutureSurfaces = nullptr;
    stream_data.pInputSurface = input_view.Get();
    stream_data.ppPastSurfacesRight = nullptr;
    stream_data.ppFutureSurfacesRight = nullptr;
    stream_data.pInputSurfaceRight = nullptr;
    
    // 执行视频处理
    HRESULT hr = video_context_->VideoProcessorBlt(video_processor_.Get(), output_view_.Get(), 0, 1, &stream_data);
    if (FAILED(hr)) {
        std::cerr << "Failed to execute video processor blt" << std::endl;
        return false;
    }
    
    // 返回 RGB 纹理引用
    rgb_texture = rgb_texture_;
    
    return true;
}

ComPtr<ID3D11VideoProcessorInputView> RgbVideoDecoder::getOrCreateInputView(ID3D11Texture2D* texture, UINT array_slice) {
    if (!texture || !video_device_ || !video_processor_enumerator_) {
        return nullptr;
    }
    
    // 创建查找键
    InputViewKey key = { texture, array_slice };
    
    // 检查缓存中是否已存在
    auto it = input_view_cache_.find(key);
    if (it != input_view_cache_.end()) {
        return it->second;
    }
    
    // 创建新的 Input View
    ComPtr<ID3D11VideoProcessorInputView> input_view;
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc = {};
    input_view_desc.FourCC = 0;
    input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    input_view_desc.Texture2D.MipSlice = 0;
    input_view_desc.Texture2D.ArraySlice = array_slice;
    
    HRESULT hr = video_device_->CreateVideoProcessorInputView(texture, video_processor_enumerator_.Get(), &input_view_desc, &input_view);
    if (FAILED(hr)) {
        std::cerr << "Failed to create video processor input view, HRESULT: 0x" << std::hex << hr << std::endl;
        return nullptr;
    }
    
    // 添加到缓存
    input_view_cache_[key] = input_view;
    
    return input_view;
}


void RgbVideoDecoder::clearInputViewCache() {
    input_view_cache_.clear();
}


void RgbVideoDecoder::cleanup() {
    clearInputViewCache();
    output_view_.Reset();
    rgb_texture_.Reset();
    video_processor_.Reset();
    video_processor_enumerator_.Reset();
    video_context_.Reset();
    video_device_.Reset();
    
    video_width_ = 0;
    video_height_ = 0;
} 