#pragma once

#include "hw_video_decoder.h"
#include <memory>
#include <string>
#include <unordered_map>

using Microsoft::WRL::ComPtr;

class RgbVideoDecoder {
public:
	struct DecodedRgbFrame {
		HwVideoDecoder::DecodedFrame hw_frame;
		ComPtr<ID3D11Texture2D> rgb_texture;
		bool is_valid;
	};
	RgbVideoDecoder();
	~RgbVideoDecoder();
	bool open(const std::string& filepath);
	bool readNextFrame(DecodedRgbFrame& frame);
	bool isOpen() const; bool isEOF() const; void close();
	HwVideoDecoder* getHwDecoder() const; int getWidth() const; int getHeight() const;
private:
	std::unique_ptr<HwVideoDecoder> hw_decoder_;
	Microsoft::WRL::ComPtr<ID3D11VideoDevice> video_device_;
	Microsoft::WRL::ComPtr<ID3D11VideoContext> video_context_;
	Microsoft::WRL::ComPtr<ID3D11VideoProcessor> video_processor_;
	Microsoft::WRL::ComPtr<ID3D11VideoProcessorEnumerator> video_processor_enumerator_;
	ComPtr<ID3D11Texture2D> rgb_texture_;
	ComPtr<ID3D11VideoProcessorOutputView> output_view_;
	struct InputViewKey { ID3D11Texture2D* texture; UINT array_slice; bool operator==(const InputViewKey& o) const { return texture==o.texture && array_slice==o.array_slice; } };
	struct InputViewKeyHash { size_t operator()(const InputViewKey& k) const { return std::hash<void*>()(k.texture) ^ (std::hash<UINT>()(k.array_slice)<<1); } };
	std::unordered_map<InputViewKey, ComPtr<ID3D11VideoProcessorInputView>, InputViewKeyHash> input_view_cache_;
	int video_width_=0; int video_height_=0;
	bool initializeVideoProcessor(); bool convertYuvToRgb(AVFrame* yuv, ComPtr<ID3D11Texture2D>& rgb); ComPtr<ID3D11VideoProcessorInputView> getOrCreateInputView(ID3D11Texture2D* tex, UINT slice); void clearInputViewCache(); void cleanup();
};
