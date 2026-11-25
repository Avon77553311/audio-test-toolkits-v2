param([switch]$SkipModels, [switch]$SkipStems)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required. Install Python 3.11 and run setup again.' }
}
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
& $pythonExe -c "import sys; assert sys.version_info[:2] == (3,11), 'Please use Python 3.11'"
if ($LASTEXITCODE -ne 0) { throw 'This environment requires Python 3.11.' }
# Replace the local shared dependency shortcut with an independent installation.
$shortcut=Join-Path $projectRoot '.venv/Lib/site-packages/full_edition_dependencies.pth'
if (Test-Path -LiteralPath $shortcut) { Remove-Item -LiteralPath $shortcut }
& $pythonExe -m pip install -r worker/requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Audio dependency installation failed.' }
& $pythonExe scripts/setup_runtime.py
if ($LASTEXITCODE -ne 0) { throw 'Native media setup failed.' }
if (-not $SkipStems) {
    & $pythonExe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
    & $pythonExe -m pip install -r worker/requirements-stems.txt
    if ($LASTEXITCODE -ne 0) { throw 'Demucs installation failed.' }
}
if (-not $SkipModels) {
    if (-not $SkipStems) { & $pythonExe scripts/prefetch_models.py }
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Rerun setup when the network is available.' }
}
Write-Output 'Local audio tools are ready. No paid services were invoked.'
