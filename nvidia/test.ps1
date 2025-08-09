# Target 写死为 integration-test
$Target = "integration-test"

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3D Player Test Script" -ForegroundColor Cyan
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

# 检查是否提供了测试文件名参数
if ($args.Length -eq 0) {
    Write-Host "Error: Test file name is required!" -ForegroundColor Red
    Write-Host "Usage: test.ps1 <test_file_name>" -ForegroundColor Yellow
    Write-Host "Example: test.ps1 '[test_mkv_stream_reader.cpp]'" -ForegroundColor Yellow
    exit 1
}

$testFile = $args[0]

# 运行指定测试
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Test: $testFile" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

try {
    & "build/Debug/integration-test.exe" $testFile -r compact
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PASSED: $testFile" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "FAILED: $testFile" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "ERROR: $testFile - $_" -ForegroundColor Red
    exit 1
}