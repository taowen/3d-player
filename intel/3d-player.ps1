param([string]$VideoFile = "", [switch]$BuildOnly = $false)
$Target = "3d_player_intel"
$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$buildDir = Join-Path $scriptDir "build"
if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
Push-Location $buildDir
try {
    Write-Host "Configuring (intel)..." -ForegroundColor Yellow
    cmake -S $scriptDir -B $buildDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Building $Target" -ForegroundColor Yellow
    cmake --build $buildDir --config Debug --target $Target
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
if ($BuildOnly) { Write-Host "Build-only. exe at intel/build/Debug/${Target}.exe" -ForegroundColor Green; exit 0 }
if ($VideoFile -and !(Test-Path $VideoFile)) { Write-Host "Video file not found: $VideoFile" -ForegroundColor Red; exit 1 }
Write-Host "Running $Target ..." -ForegroundColor Cyan
$exe = Join-Path $buildDir "Debug/${Target}.exe"
if ($VideoFile) { & $exe $VideoFile } else { & $exe }
exit $LASTEXITCODE
