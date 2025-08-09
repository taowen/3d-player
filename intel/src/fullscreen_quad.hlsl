/**
 * @file fullscreen_quad.hlsl
 * @brief 全屏四边形渲染 shader，用于将视频纹理渲染到屏幕
 * 
 * 使用无顶点缓冲区技术：通过 SV_VertexID 生成全屏三角形
 * 支持纹理采样和基本的像素着色
 */

// 纹理和采样器
Texture2D videoTexture : register(t0);
SamplerState textureSampler : register(s0);


// Vertex Shader 输出结构
struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};


/**
 * @brief 顶点着色器 - 生成全屏三角形
 * @param vertexID 顶点 ID (0, 1, 2)
 * @return VSOutput 顶点着色器输出
 * 
 * 使用无顶点缓冲区技术：
 * - 顶点 0: (-1, -3) -> 屏幕左下角外
 * - 顶点 1: (-1,  1) -> 屏幕左上角
 * - 顶点 2: ( 3,  1) -> 屏幕右上角外
 * 
 * 这个大三角形覆盖整个屏幕，GPU 会自动裁剪多余部分
 */
VSOutput VSMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    
    // 生成全屏三角形的顶点位置
    float2 texCoord = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(texCoord * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    output.texCoord = texCoord;
    
    return output;
}


/**
 * @brief 像素着色器 - 采样视频纹理并输出颜色
 * @param input 顶点着色器输出
 * @return float4 最终像素颜色
 * 
 * 简单的纹理采样，支持线性过滤用于缩放
 */
float4 PSMain(VSOutput input) : SV_Target {
    // 采样视频纹理
    float4 color = videoTexture.Sample(textureSampler, input.texCoord);
    
    // 直接输出颜色（可以在这里添加色彩校正、亮度调整等）
    return color;
} 