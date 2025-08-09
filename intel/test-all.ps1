$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$buildDir = Join-Path $scriptDir "build"
if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
Push-Location $buildDir
try {
  cmake -S $scriptDir -B $buildDir
  cmake --build $buildDir --config Debug --target integration-test-intel
} finally { Pop-Location }
& (Join-Path $buildDir "Debug/integration-test-intel.exe")
