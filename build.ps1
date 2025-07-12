param(
    [string]$Target = "test"
)

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
    cmake -S .. -B .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed! Exiting..." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    # 根据参数设置构建目标和描述
    $BuildTarget = ""
    $BuildDesc = ""
    $RunTests = $false

    switch ($Target.ToLower()) {
        "test" {
            $BuildTarget = "integration-test"
            $BuildDesc = "test target"
            $RunTests = $true
        }
        "3d_player" {
            $BuildTarget = "3d_player"
            $BuildDesc = "3d_player target"
            $RunTests = $false
        }
        default {
            Write-Host "Unknown target: $Target" -ForegroundColor Red
            Write-Host ""
            Write-Host "Usage: .\build.ps1 [target]" -ForegroundColor White
            Write-Host "Available targets:" -ForegroundColor White
            Write-Host "  test       - Build and run tests (default)" -ForegroundColor White
            Write-Host "  3d_player  - Build 3d_player executable" -ForegroundColor White
            exit 1
        }
    }

    # 执行构建
    Write-Host "Building $BuildDesc..." -ForegroundColor Yellow
    if ($BuildTarget -eq "") {
        cmake --build . --config Debug
    } else {
        cmake --build . --config Debug --target $BuildTarget
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed! Exiting..." -ForegroundColor Red
        exit $LASTEXITCODE
    }

} finally {
    # 返回原目录
    Pop-Location
}

Write-Host ""

# 根据目标执行后续操作
if ($RunTests) {
    Write-Host "Running tests..." -ForegroundColor Green
    & "build\Debug\integration-test.exe" --success --verbosity high
} else {
    Write-Host "Build completed successfully!" -ForegroundColor Green
    if ($Target.ToLower() -eq "3d_player") {
        Write-Host ""
        Write-Host "Usage: .\build\Debug\3d_player.exe <video_file_path>" -ForegroundColor White
        Write-Host "Example: .\build\Debug\3d_player.exe test_data\sample_hw.mkv" -ForegroundColor White
    }
} 