#include <windows.h>
#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include "audio_video_player.h"

using Microsoft::WRL::ComPtr;

// 窗口过程函数声明
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

// D3D11 初始化函数声明
bool createSwapChain(HWND hwnd);
void cleanupD3D11();
bool initializeAudioVideoPlayer(const std::string& videoFilePath);
bool finalizeAudioVideoPlayer();

// 全局变量
const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;
const wchar_t* WINDOW_CLASS_NAME = L"3DPlayerWindow";
const wchar_t* WINDOW_TITLE = L"3D Player";

// D3D11 设备和交换链
ComPtr<ID3D11Device> g_device;
ComPtr<ID3D11DeviceContext> g_context;
ComPtr<IDXGISwapChain> g_swapchain;

// AudioVideoPlayer 实例
std::unique_ptr<AudioVideoPlayer> g_audio_video_player;

// 时间管理
std::chrono::high_resolution_clock::time_point g_start_time;


/**
 * @brief 显示使用说明
 */
void showUsage() {
    std::cout << "Usage: 3d_player.exe <video_file_path>" << std::endl;
    std::cout << "Example: 3d_player.exe test_data/sample_hw.mkv" << std::endl;
}


/**
 * @brief 将多字节字符串转换为宽字符串
 * @param str 多字节字符串
 * @return std::wstring 宽字符串
 */
std::wstring stringToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
    
    return wstr;
}


/**
 * @brief 注册窗口类
 * @param hInstance 应用程序实例句柄
 * @return bool 成功返回 true，失败返回 false
 */
bool registerWindowClass(HINSTANCE hInstance) {
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(WNDCLASSEXW);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName = NULL;
    wc.lpszClassName = WINDOW_CLASS_NAME;
    wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

    return RegisterClassExW(&wc) != 0;
}


/**
 * @brief 创建窗口
 * @param hInstance 应用程序实例句柄
 * @param videoFilePath 视频文件路径（用于窗口标题）
 * @return HWND 窗口句柄，失败返回 NULL
 */
HWND createWindow(HINSTANCE hInstance, const std::string& videoFilePath) {
    // 创建窗口标题
    std::wstring title = WINDOW_TITLE;
    if (!videoFilePath.empty()) {
        title += L" - " + stringToWString(videoFilePath);
    }
    
    // 计算窗口大小（包含边框）
    RECT windowRect = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };
    AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);
    
    int windowWidth = windowRect.right - windowRect.left;
    int windowHeight = windowRect.bottom - windowRect.top;
    
    // 获取屏幕中心位置
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    int posX = (screenWidth - windowWidth) / 2;
    int posY = (screenHeight - windowHeight) / 2;
    
    HWND hwnd = CreateWindowExW(
        0,                              // Extended window style
        WINDOW_CLASS_NAME,              // Window class name
        title.c_str(),                  // Window title
        WS_OVERLAPPEDWINDOW,            // Window style
        posX, posY,                     // Position
        windowWidth, windowHeight,      // Size
        NULL,                           // Parent window
        NULL,                           // Menu
        hInstance,                      // Instance handle
        NULL                            // Additional application data
    );
    
    return hwnd;
}


/**
 * @brief 窗口过程函数
 * @param hwnd 窗口句柄
 * @param uMsg 消息标识符
 * @param wParam 消息参数
 * @param lParam 消息参数
 * @return LRESULT 消息处理结果
 */
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
            
        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;
            
        case WM_PAINT: {
            PAINTSTRUCT ps;
            BeginPaint(hwnd, &ps);
            // D3D11 负责渲染，这里不需要做任何事情
            EndPaint(hwnd, &ps);
            return 0;
        }
        
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                PostQuitMessage(0);
                return 0;
            }
            break;
    }
    
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


/**
 * @brief 创建交换链（使用现有的 D3D11 设备）
 * @param hwnd 窗口句柄
 * @return bool 成功返回 true，失败返回 false
 */
bool createSwapChain(HWND hwnd) {
    if (!g_device) {
        std::cerr << "错误：D3D11 设备未初始化" << std::endl;
        return false;
    }
    
    // 创建交换链
    DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
    swapChainDesc.BufferCount = 2;
    swapChainDesc.BufferDesc.Width = WINDOW_WIDTH;
    swapChainDesc.BufferDesc.Height = WINDOW_HEIGHT;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = hwnd;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.Windowed = TRUE;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.Flags = 0;
    
    ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT hr = g_device.As(&dxgiDevice);
    if (FAILED(hr)) {
        std::cerr << "错误：获取 DXGI 设备失败" << std::endl;
        return false;
    }
    
    ComPtr<IDXGIAdapter> dxgiAdapter;
    hr = dxgiDevice->GetAdapter(&dxgiAdapter);
    if (FAILED(hr)) {
        std::cerr << "错误：获取 DXGI 适配器失败" << std::endl;
        return false;
    }
    
    ComPtr<IDXGIFactory> dxgiFactory;
    hr = dxgiAdapter->GetParent(IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) {
        std::cerr << "错误：获取 DXGI 工厂失败" << std::endl;
        return false;
    }
    
    hr = dxgiFactory->CreateSwapChain(g_device.Get(), &swapChainDesc, &g_swapchain);
    if (FAILED(hr)) {
        std::cerr << "错误：创建交换链失败 (HRESULT: 0x" << std::hex << hr << ")" << std::endl;
        return false;
    }
    
    std::cout << "交换链创建成功" << std::endl;
    return true;
}


/**
 * @brief 清理 D3D11 资源
 */
void cleanupD3D11() {
    if (g_audio_video_player) {
        g_audio_video_player->close();
        g_audio_video_player.reset();
    }
    
    g_swapchain.Reset();
    g_context.Reset();
    g_device.Reset();
    
    std::cout << "D3D11 资源清理完成" << std::endl;
}


/**
 * @brief 初始化 AudioVideoPlayer（第一步：打开文件获取设备）
 * @param videoFilePath 视频文件路径
 * @return bool 成功返回 true，失败返回 false
 */
bool initializeAudioVideoPlayer(const std::string& videoFilePath) {
    g_audio_video_player = std::make_unique<AudioVideoPlayer>();
    
    // 第一步：打开音视频文件（获取 D3D11 设备）
    if (!g_audio_video_player->open(videoFilePath)) {
        std::cerr << "错误：打开音视频文件失败: " << videoFilePath << std::endl;
        return false;
    }
    
    // 获取 AudioVideoPlayer 创建的设备
    g_device = g_audio_video_player->getD3D11Device();
    g_context = g_audio_video_player->getD3D11DeviceContext();
    
    if (!g_device || !g_context) {
        std::cerr << "错误：获取 D3D11 设备失败" << std::endl;
        return false;
    }
    
    // 添加引用计数
    g_device->AddRef();
    g_context->AddRef();
    
    std::cout << "AudioVideoPlayer 初始化成功" << std::endl;
    return true;
}


/**
 * @brief 完成 AudioVideoPlayer 的初始化（第二步：设置交换链渲染目标和音频）
 * @return bool 成功返回 true，失败返回 false
 */
bool finalizeAudioVideoPlayer() {
    if (!g_audio_video_player || !g_swapchain) {
        std::cerr << "错误：AudioVideoPlayer 或交换链未初始化" << std::endl;
        return false;
    }
    
    // 第二步：设置交换链渲染目标
    if (!g_audio_video_player->setRenderTarget(g_swapchain)) {
        std::cerr << "错误：设置交换链渲染目标失败" << std::endl;
        return false;
    }
    
    // 第三步：初始化音频播放器（如果有音频流）
    if (g_audio_video_player->hasAudio()) {
        if (!g_audio_video_player->initializeAudio()) {
            std::cerr << "警告：音频初始化失败，仅播放视频" << std::endl;
        } else {
            std::cout << "音频初始化成功" << std::endl;
        }
    }
    
    std::cout << "AudioVideoPlayer 完整初始化成功" << std::endl;
    return true;
}


/**
 * @brief 主函数
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return int 程序退出代码
 */
int main(int argc, char* argv[]) {
    // 设置控制台代码页为UTF-8，解决中文显示乱码问题
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    // 初始化 COM 库用于 WASAPI 音频播放
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::cerr << "错误：COM 库初始化失败: " << std::hex << hr << std::endl;
        return 1;
    }
    
    // 检查命令行参数
    if (argc < 2) {
        std::cerr << "错误：缺少视频文件路径参数" << std::endl;
        showUsage();
        CoUninitialize();
        return 1;
    }
    
    std::string videoFilePath = argv[1];
    std::cout << "视频文件路径: " << videoFilePath << std::endl;
    
    // 获取应用程序实例句柄
    HINSTANCE hInstance = GetModuleHandle(NULL);
    if (!hInstance) {
        std::cerr << "错误：无法获取应用程序实例句柄" << std::endl;
        CoUninitialize();
        return 1;
    }
    
    // 注册窗口类
    if (!registerWindowClass(hInstance)) {
        std::cerr << "错误：注册窗口类失败" << std::endl;
        CoUninitialize();
        return 1;
    }
    
    // 创建窗口
    HWND hwnd = createWindow(hInstance, videoFilePath);
    if (!hwnd) {
        std::cerr << "错误：创建窗口失败" << std::endl;
        CoUninitialize();
        return 1;
    }
    
    // 显示窗口
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);
    
    std::cout << "窗口创建成功！" << std::endl;
    
    // 第一步：初始化 AudioVideoPlayer（获取 D3D11 设备）
    if (!initializeAudioVideoPlayer(videoFilePath)) {
        std::cerr << "错误：AudioVideoPlayer 初始化失败" << std::endl;
        CoUninitialize();
        return 1;
    }
    
    // 第二步：创建交换链（使用 AudioVideoPlayer 的设备）
    if (!createSwapChain(hwnd)) {
        std::cerr << "错误：创建交换链失败" << std::endl;
        cleanupD3D11();
        CoUninitialize();
        return 1;
    }
    
    // 第三步：完成 AudioVideoPlayer 初始化（设置交换链和音频）
    if (!finalizeAudioVideoPlayer()) {
        std::cerr << "错误：AudioVideoPlayer 完整初始化失败" << std::endl;
        cleanupD3D11();
        CoUninitialize();
        return 1;
    }
    
    std::cout << "所有组件初始化成功！开始播放音视频..." << std::endl;
    std::cout << "按 ESC 键退出" << std::endl;
    
    // 记录开始时间
    g_start_time = std::chrono::high_resolution_clock::now();
    
    // 消息循环
    MSG msg = {};
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            // 计算相对时间
            auto current_time = std::chrono::high_resolution_clock::now();
            double relative_time = std::chrono::duration<double>(current_time - g_start_time).count();
            
            // 更新音视频播放
            if (g_audio_video_player) {
                g_audio_video_player->onTimer(relative_time);
                
                // 检查是否播放结束
                if (g_audio_video_player->isEOF()) {
                    std::cout << "音视频播放结束" << std::endl;
                    PostQuitMessage(0);
                }
            }
            
            // 注意：AudioVideoPlayer 内部已经处理了交换链的 Present 操作
            // 不需要在这里再次调用 Present
            
            // 让出CPU时间片
            Sleep(1);
        }
    }
    
    std::cout << "程序退出" << std::endl;
    cleanupD3D11();
    CoUninitialize();
    return 0;
} 