$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$serverRoot = Join-Path $projectRoot 'server'
& mvn -q -s (Join-Path $serverRoot 'maven-settings.xml') -f (Join-Path $serverRoot 'pom.xml') "-Dmaven.repo.local=$projectRoot/.local-tools/m2-central" -DskipTests package
if ($LASTEXITCODE -ne 0) { throw 'Java business server build failed. Install JDK 17+ and Maven, then retry.' }
Copy-Item -LiteralPath (Join-Path $serverRoot 'build/maven/audio-studio-server.jar') -Destination (Join-Path $serverRoot 'audio-studio-server.jar') -Force
Write-Output 'Local Java server built successfully.'
