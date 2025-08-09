$Target = "integration-test-intel"
$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$buildDir = Join-Path $scriptDir "build"
if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
Push-Location $buildDir
try {
  cmake -S $scriptDir -B $buildDir
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  cmake --build $buildDir --config Debug --target $Target
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
if ($args.Length -eq 0) { Write-Host "Need test tag e.g. '[test_mkv_stream_reader.cpp]'" -ForegroundColor Red; exit 1 }
& (Join-Path $buildDir "Debug/${Target}.exe") $args[0] -r compact
exit $LASTEXITCODE
