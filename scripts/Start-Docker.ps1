param([string]$EnvFile='deploy.env',[ValidatePattern('^audio-toolkit-short(?:-[a-z0-9]+)*$')][string]$ProjectName='audio-toolkit-short')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
foreach ($path in @('dist/index.html','server/audio-studio-server.jar','worker/app.py')) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {throw "Missing build output: $path"}
}
foreach ($path in @('runtime/models','runtime/torch/hub/checkpoints')) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {throw "Missing models: $path. See README.md"}
}
if (-not (Test-Path -LiteralPath $EnvFile)) {throw 'Copy deploy/deploy.env.example to deploy.env and set the target IP and port first.'}
# A bounded readiness check prevents docker info from looking frozen indefinitely.
$probe=Start-Process -FilePath 'docker' -ArgumentList @('info','--format','{{.OSType}}') -WindowStyle Hidden -PassThru -RedirectStandardOutput '.local-docker-info.tmp' -RedirectStandardError '.local-docker-error.tmp'
# Cache the process handle before waiting (Windows PowerShell 5.1 ExitCode reliability).
$probeHandle=$probe.Handle
if (-not $probe.WaitForExit(20000)) {Stop-Process -Id $probe.Id -Force;throw 'Docker did not respond within 20 seconds. Start Docker Desktop and wait for Engine running.'}
$probe.Refresh()
if ($probe.ExitCode -ne 0 -or (Get-Content -LiteralPath '.local-docker-info.tmp' -Raw).Trim() -ne 'linux') {throw 'Docker Desktop must be running in Linux containers mode.'}
Remove-Item -LiteralPath '.local-docker-info.tmp','.local-docker-error.tmp'
& docker compose -p $ProjectName --env-file $EnvFile -f compose.yaml config --quiet
if ($LASTEXITCODE -ne 0) {throw 'Compose validation failed'}
& docker compose -p $ProjectName --env-file $EnvFile -f compose.yaml up -d --force-recreate --no-build --pull never
if ($LASTEXITCODE -ne 0) {throw 'Startup failed. See the Docker error above; check the configured port and import the shared runtime image if missing.'}
& docker compose -p $ProjectName --env-file $EnvFile -f compose.yaml ps
Write-Output 'Audio Toolkit Short started. Use the IP and port in deploy.env. No full-edition volume or container was changed.'
