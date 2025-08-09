# 设置错误处理
$ErrorActionPreference = "Stop"

# 创建build目录
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

# 进入build目录
Push-Location "build"

try {
    # 配置和构建
    cmake -S .. -B .
    cmake --build . --config Debug --target integration-test
} finally {
    Pop-Location
}

# 运行所有测试
& "build/Debug/integration-test.exe"