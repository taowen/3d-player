# 3d-player.ps1
# 3D Player构建和运行脚本

param(
    [string]$VideoFile = "",
    [switch]$BuildOnly = $false
)

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

# 如果指定了 BuildOnly，则只编译不运行
if ($BuildOnly) {
    Write-Host "Build-only mode. Executable created at: build/Debug/3d_player.exe" -ForegroundColor Yellow
    exit 0
}

# 运行程序
Write-Host "" -ForegroundColor White
Write-Host "Running 3D Player..." -ForegroundColor Yellow

# 如果提供了视频文件参数，检查文件是否存在
if ($VideoFile -ne "") {
    Write-Host "  Video file: $VideoFile" -ForegroundColor White
    Write-Host "" -ForegroundColor White
    
    # 检查视频文件是否存在
    if (!(Test-Path $VideoFile)) {
        Write-Host "Error: Video file '$VideoFile' not found" -ForegroundColor Red
        Write-Host "Available test files:" -ForegroundColor Yellow
        if (Test-Path "test_data") {
            Get-ChildItem "test_data" -Filter "*.mkv" | ForEach-Object { Write-Host "  test_data/$($_.Name)" -ForegroundColor White }
        }
        exit 1
    }
    
    # 运行程序并传入视频文件参数
    Write-Host "Executing: build/Debug/3d_player.exe $VideoFile" -ForegroundColor Cyan
    & "build/Debug/3d_player.exe" $VideoFile
} else {
    # 没有参数时运行程序
    Write-Host "Executing: build/Debug/3d_player.exe" -ForegroundColor Cyan
    & "build/Debug/3d_player.exe"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "" -ForegroundColor White
    Write-Host "=== 3D Player execution completed! ===" -ForegroundColor Green
} else {
    Write-Host "" -ForegroundColor White
    Write-Host "Execution failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}