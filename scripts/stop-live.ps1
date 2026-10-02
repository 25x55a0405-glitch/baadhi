<#
.SYNOPSIS
  Take the complete Baadhi offline again: stop the tunnel, the server and the keep-awake helper, and send visitors
  of https://baadhi.pages.dev back to the saved-analysis demo at once.
#>
param([switch]$Quiet)
$root = Split-Path -Parent $PSScriptRoot
$state = Join-Path $root ".work\live"
$env:WRANGLER_SEND_METRICS = "false"
$f = Join-Path $state "live.json"

function Stop-Mine([object]$id, [string]$pattern) {
  # only stop the process if it is still the one we started (the id could have been reused)
  if (-not $id) { return }
  $p = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
  if ($p -and $p.CommandLine -match $pattern) { & taskkill /PID $id /T /F 2>&1 | Out-Null }
}

if (Test-Path $f) {
  $j = Get-Content $f -Raw | ConvertFrom-Json
  Stop-Mine $j.tunnel 'wrangler|cloudflared'
  Stop-Mine $j.server 'serve\.py'
  Stop-Mine $j.keepawake 'keep-awake\.ps1'
  Remove-Item $f -ErrorAction SilentlyContinue
}

# visitors get the demo immediately (the worker would notice within seconds anyway)
$toml = Join-Path $root "wrangler.toml"
if (Test-Path $toml) {
  $kv = (Select-String -Path $toml -Pattern '^id\s*=\s*"([0-9a-f]+)"').Matches[0].Groups[1].Value
  try { wrangler kv key delete --namespace-id $kv --remote "origin" 2>&1 | Out-Null } catch { }
}
if (-not $Quiet) { Write-Host "Offline. https://baadhi.pages.dev now shows the saved-analysis demo." }
