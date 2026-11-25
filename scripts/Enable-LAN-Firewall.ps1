param([ValidateRange(1,65535)][int]$Port = 8081)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$receiptPath = Join-Path $projectRoot 'runtime/lan-firewall-status.json'
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'runtime') | Out-Null
trap {
    @{success=$false;message=$_.Exception.Message} | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding UTF8
    exit 1
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in an Administrator PowerShell window to allow LAN access.'
}
$javaCommand = (Get-Command java -ErrorAction Stop).Source
if ($env:JAVA_HOME -and (Test-Path -LiteralPath "$env:JAVA_HOME\bin\java.exe")) {
    $javaCommand = Join-Path $env:JAVA_HOME 'bin/java.exe'
}
$ruleName = "AudioToolkit-LAN-TCP-$Port"
$addresses = @(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
    $_.AddressState -eq 'Preferred' -and ($_.IPAddress -match '^10\.' -or $_.IPAddress -match '^192\.168\.' -or $_.IPAddress -match '^172\.(1[6-9]|2[0-9]|3[01])\.')
} | Select-Object -ExpandProperty IPAddress)
if (-not $addresses.Count) { throw 'No private IPv4 interface was found.' }
$existing = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    $existing | Set-NetFirewallRule -Enabled True -Profile Any -Action Allow -Direction Inbound -Protocol TCP -LocalPort $Port -LocalAddress $addresses -RemoteAddress LocalSubnet -Program $javaCommand | Out-Null
} else {
    New-NetFirewallRule -Name $ruleName -DisplayName "Audio Toolkit Short LAN TCP $Port" -Enabled True -Direction Inbound -Action Allow -Profile Any -Protocol TCP -LocalPort $Port -LocalAddress $addresses -RemoteAddress LocalSubnet -Program $javaCommand | Out-Null
}
Write-Output "Allowed Java TCP $Port on private interface addresses, for LocalSubnet peers only."
@{success=$true;port=$Port;addresses=$addresses;program=$javaCommand;remote='LocalSubnet'} | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding UTF8
