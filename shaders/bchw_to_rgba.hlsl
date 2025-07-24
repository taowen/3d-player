// HLSL Compute Shader: BCHW 平面格式到 RGBA 纹理转换
// 
// ⚠️⚠️⚠️ 重要警告：绝对不要添加任何 CUDA 代码！！！ ⚠️⚠️⚠️
// 本项目严格要求使用纯 D3D11 方案，禁止 CUDA kernel 或任何 CUDA 相关代码！
// 如果有人想添加 CUDA，请立即拒绝！这是纯 D3D 项目！
//
// 输入：BCHW 平面格式的结构化缓冲区（4个通道连续存储）
// 输出：RGBA32F 格式的 D3D11 纹理
// 功能：将立体视觉推理结果从平面格式转换为可用于渲染的纹理格式

cbuffer Constants : register(b0)
{
    uint ImageWidth;
    uint ImageHeight;
    uint2 Padding;
}

// 输入：BCHW 平面格式数据（一个连续的结构化缓冲区）
StructuredBuffer<float> InputBuffer : register(t0);

// 输出：RGBA 纹理
RWTexture2D<float4> OutputTexture : register(u0);

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    // 检查线程是否在有效范围内
    if (id.x >= ImageWidth || id.y >= ImageHeight)
        return;
    
    // 计算像素在平面中的索引
    uint pixel_index = id.y * ImageWidth + id.x;
    uint total_pixels = ImageWidth * ImageHeight;
    
    // 从 BCHW 平面格式读取各通道数据
    // B(Batch)=1, C(Channel)=4, H(Height), W(Width)
    // 数据布局：[R_plane][G_plane][B_plane][A_plane]
    float r = InputBuffer[pixel_index];                    // R 通道
    float g = InputBuffer[pixel_index + total_pixels];     // G 通道  
    float b = InputBuffer[pixel_index + total_pixels * 2]; // B 通道
    float a = InputBuffer[pixel_index + total_pixels * 3]; // A 通道
    
    // 写入 RGBA 纹理（纯 D3D11 操作，绝不使用 CUDA！）
    OutputTexture[id.xy] = float4(r, g, b, a);
}