# stereo_video_decoder_cli.ps1
# 立体视频解码器命令行工具构建和运行脚本

param(
    [string]$VideoFile = "test_data/sample_hw.mkv",
    [int]$FrameNumber = 5,
    [switch]$BuildOnly = $false
)

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "=== Stereo Video Decoder CLI ===" -ForegroundColor Green

# 构建步骤
Write-Host "Building stereo_video_decoder_cli..." -ForegroundColor Yellow

# 配置 CMake
Write-Host "Configuring CMake..." -ForegroundColor Cyan
cmake -B build -DCMAKE_BUILD_TYPE=Debug
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed" -ForegroundColor Red
    exit 1
}

# 编译目标
Write-Host "Compiling stereo_video_decoder_cli..." -ForegroundColor Cyan
cmake --build build --config Debug --target stereo_video_decoder_cli
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "Build completed successfully!" -ForegroundColor Green

# 如果指定了 BuildOnly，则只编译不运行
if ($BuildOnly) {
    Write-Host "Build-only mode. Executable created at: build/Debug/stereo_video_decoder_cli.exe" -ForegroundColor Yellow
    exit 0
}

# 运行程序
Write-Host "" -ForegroundColor White
Write-Host "Running stereo_video_decoder_cli with:" -ForegroundColor Yellow
Write-Host "  Video file: $VideoFile" -ForegroundColor White
Write-Host "  Frame number: $FrameNumber" -ForegroundColor White
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

# 运行程序
Write-Host "Executing: build/Debug/stereo_video_decoder_cli.exe $VideoFile $FrameNumber" -ForegroundColor Cyan
& "build/Debug/stereo_video_decoder_cli.exe" $VideoFile $FrameNumber

if ($LASTEXITCODE -eq 0) {
    Write-Host "" -ForegroundColor White
    Write-Host "=== Execution completed successfully! ===" -ForegroundColor Green
    Write-Host "Generated files:" -ForegroundColor Yellow
    
    # 显示生成的文件
    $inputFile = "frame_${FrameNumber}_input.bmp"
    $outputFile = "frame_${FrameNumber}_output.bmp"
    
    if (Test-Path $inputFile) {
        Write-Host "  $inputFile - Input RGB frame" -ForegroundColor White
    }
    if (Test-Path $outputFile) {
        Write-Host "  $outputFile - Stereo output frame" -ForegroundColor White
    }
} else {
    Write-Host "" -ForegroundColor White
    Write-Host "Execution failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}