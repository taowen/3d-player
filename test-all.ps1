# Target 写死为 integration-test
$Target = "integration-test"

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3D Player Test All Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 创建build目录
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

# 进入build目录
Push-Location "build"

try {
    # 配置项目
    Write-Host "Configuring with CMake..." -ForegroundColor Yellow
    Write-Host "AddressSanitizer always enabled" -ForegroundColor Magenta
    $cmakeArgs = @("-S", "..", "-B", ".")
    cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed! Exiting..." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    # 执行构建
    Write-Host "Building $Target..." -ForegroundColor Yellow
    cmake --build . --config Debug --target $Target

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed! Exiting..." -ForegroundColor Red
        exit $LASTEXITCODE
    }

} finally {
    # 返回原目录
    Pop-Location
}

Write-Host ""
Write-Host "Build completed successfully!" -ForegroundColor Green

# 获取所有测试文件
$testFiles = @(
    "[test_mkv_stream_reader.cpp]",
    "[test_hw_video_decoder.cpp]", 
    "[test_audio_decoder.cpp]",
    "[test_rgb_video_decoder.cpp]",
    "[test_video_player.cpp]",
    "[test_audio_player.cpp]",
    "[test_audio_video_player.cpp]",
    "[test_tensorrt.cpp]",
    "[test_d3d11_cuda_interop.cpp]",
    "[test_stereo_video_decoder.cpp]"
)

# 运行所有测试
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running All Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$passedTests = 0
$failedTests = 0
$failedTestNames = @()

foreach ($testFile in $testFiles) {
    Write-Host ""
    Write-Host "Running: $testFile" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Gray
    
    try {
        & "build/Debug/integration-test.exe" $testFile -r compact
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ PASSED: $testFile" -ForegroundColor Green
            $passedTests++
        } else {
            Write-Host "✗ FAILED: $testFile" -ForegroundColor Red
            $failedTests++
            $failedTestNames += $testFile
        }
    } catch {
        Write-Host "✗ ERROR: $testFile - $_" -ForegroundColor Red
        $failedTests++
        $failedTestNames += $testFile
    }
}

# 输出测试总结
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total Tests: $($testFiles.Length)" -ForegroundColor White
Write-Host "Passed: $passedTests" -ForegroundColor Green
Write-Host "Failed: $failedTests" -ForegroundColor Red

if ($failedTests -gt 0) {
    Write-Host ""
    Write-Host "Failed Tests:" -ForegroundColor Red
    foreach ($failedTest in $failedTestNames) {
        Write-Host "  - $failedTest" -ForegroundColor Red
    }
    exit 1
} else {
    Write-Host ""
    Write-Host "🎉 All tests passed!" -ForegroundColor Green
    exit 0
}