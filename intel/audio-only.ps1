param(
    [string]$AudioFile,
    [switch]$BuildOnly
)

$ErrorActionPreference = "Stop"

# Check if in intel directory
if (!(Test-Path "CMakeLists.txt") -or !(Test-Path "src")) {
    Write-Error "Please run this script in intel directory"
    exit 1
}

Write-Host "Building audio player..." -ForegroundColor Green

# Create build directory
if (!(Test-Path "build")) {
    New-Item -ItemType Directory -Name "build" | Out-Null
}

# Enter build directory and build
Push-Location "build"

try {
    # Configure CMake
    cmake -G "Visual Studio 17 2022" -A x64 ..
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configuration failed"
        exit 1
    }

    # Build project
    cmake --build . --config Debug --target audio_only_player
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed"
        exit 1
    }

    Write-Host "Build successful!" -ForegroundColor Green

    # If build only, exit
    if ($BuildOnly) {
        Write-Host "Build only mode, skipping run" -ForegroundColor Yellow
        exit 0
    }

    # Run audio player
    if ($AudioFile) {
        Write-Host "Running audio player: $AudioFile" -ForegroundColor Cyan
        & ".\Debug\audio_only_player.exe" $AudioFile
    } else {
        Write-Host "Running audio player (default file)" -ForegroundColor Cyan
        & ".\Debug\audio_only_player.exe" "..\test_data\sample_with_audio.mkv"
    }

} finally {
    Pop-Location
}