# Target 写死为 3d_player
$Target = "3d_player"

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3D Player Build Script" -ForegroundColor Cyan
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