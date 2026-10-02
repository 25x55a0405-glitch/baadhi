<#
.SYNOPSIS
  Put the COMPLETE Baadhi online: run the real server on this laptop and open a Cloudflare tunnel to it.

.DESCRIPTION
  While this is running, https://baadhi.pages.dev serves the complete dashboard (analyse any area and date, trace flood
  paths); when it is not, the same address shows the saved-analysis demo. Stop it with scripts\stop-live.ps1.

  What happens, in order:
    1. the dashboard server starts in public mode (low priority, so the laptop stays usable); visitors get a small
       allowance (3 analyses per hour each, short waiting line) — your own browser on this laptop is never limited;
    2. Cloudflare's quick tunnel (wrangler tunnel quick-start) gives it a temporary https address;
    3. that address is written to Cloudflare KV, where the Pages worker finds it (see experiments/pages_worker.js);
    4. the laptop is kept awake (unless -NoKeepAwake) — a sleeping laptop would drop the connection.

  Needs: the project's .venv, wrangler logged in (wrangler login), and the Pages site deployed once.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\go-live.ps1
#>
param([switch]$NoKeepAwake)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$state = Join-Path $root ".work\live"
New-Item -ItemType Directory -Force $state | Out-Null
$env:WRANGLER_SEND_METRICS = "false"
if (-not (Test-Path $py)) { throw "Missing $py - create the project's .venv first (see README, Setup)." }
$kv = (Select-String -Path (Join-Path $root "wrangler.toml") -Pattern '^id\s*=\s*"([0-9a-f]+)"').Matches[0].Groups[1].Value

Write-Host "1/4 Clearing any earlier live session ..."
& (Join-Path $PSScriptRoot "stop-live.ps1") -Quiet

# a secret shared with the Cloudflare worker, so the server can trust the visitor address the worker relays
$tokenFile = Join-Path $state "edge_token.txt"
if (-not (Test-Path $tokenFile)) {
  $b = New-Object byte[] 24
  [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
  ([Convert]::ToBase64String($b) -replace '[+/=]', 'x') | Set-Content -Path $tokenFile -NoNewline
}
$token = (Get-Content $tokenFile -Raw).Trim()

Write-Host "2/4 Starting the dashboard server in public mode ..."
$env:BAADHI_PUBLIC = "1"
$env:BAADHI_EDGE_TOKEN = $token
$srvLog = Join-Path $state "server.log"
$srv = Start-Process -FilePath $py -ArgumentList @("serve.py") -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $srvLog -RedirectStandardError (Join-Path $state "server.err.log")
try { $srv.PriorityClass = "BelowNormal" } catch { }
$port = $null
for ($i = 0; $i -lt 90 -and -not $port; $i++) {
  Start-Sleep -Seconds 1
  if (Test-Path $srvLog) {
    $m = Select-String -Path $srvLog -Pattern 'Baadhi dashboard on http://127\.0\.0\.1:(\d+)' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { $port = [int]$m.Matches[0].Groups[1].Value }
  }
  if ($srv.HasExited) { throw "The server stopped at start-up - see $state\server.err.log" }
}
if (-not $port) { throw "The server did not report a port - see $srvLog" }
$ok = $false
for ($i = 0; $i -lt 60 -and -not $ok; $i++) {
  try { $h = Invoke-RestMethod "http://127.0.0.1:$port/api/health" -TimeoutSec 3; $ok = ($h.app -eq "baadhi" -and $h.public -eq $true) } catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) { throw "The server is not answering in public mode on port $port." }

Write-Host "3/4 Opening the Cloudflare tunnel (the first time, wrangler downloads Cloudflare's tunnel program) ..."
$tunLog = Join-Path $state "tunnel.log"
$tunErr = Join-Path $state "tunnel.err.log"
$tun = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "wrangler tunnel quick-start http://127.0.0.1:$port") -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $tunLog -RedirectStandardError $tunErr
$url = $null
for ($i = 0; $i -lt 180 -and -not $url; $i++) {
  Start-Sleep -Seconds 1
  foreach ($f in @($tunLog, $tunErr)) {
    if (Test-Path $f) {
      $m = Select-String -Path $f -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($m) { $url = $m.Matches[0].Value; break }
    }
  }
  if ($tun.HasExited -and -not $url) { throw "The tunnel stopped - see $tunLog and $tunErr" }
}
if (-not $url) { throw "No tunnel address appeared - see $tunLog and $tunErr" }
$ok = $false
for ($i = 0; $i -lt 90 -and -not $ok; $i++) {
  try { $h = Invoke-RestMethod "$url/api/health" -TimeoutSec 5; $ok = ($h.app -eq "baadhi") } catch { Start-Sleep -Seconds 2 }
}
if (-not $ok) { throw "The tunnel address $url does not answer yet - try again in a minute." }

Write-Host "4/4 Telling baadhi.pages.dev where the live engine is ..."
wrangler kv key put --namespace-id $kv --remote "edge_token" $token | Out-Null
wrangler kv key put --namespace-id $kv --remote "origin" $url | Out-Null

$ka = $null
if (-not $NoKeepAwake) {
  $ka = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", (Join-Path $PSScriptRoot "keep-awake.ps1")) -WindowStyle Hidden -PassThru
}
@{ server = $srv.Id; tunnel = $tun.Id; keepawake = $(if ($ka) { $ka.Id } else { $null }); port = $port; url = $url; started = (Get-Date).ToString("s") } |
  ConvertTo-Json | Set-Content -Path (Join-Path $state "live.json")

Write-Host ""
Write-Host "LIVE  https://baadhi.pages.dev   (relayed to $url)" -ForegroundColor Green
Write-Host "      Your own browser:  http://127.0.0.1:$port   (no visitor limits)"
Write-Host "      Stop:              powershell -ExecutionPolicy Bypass -File scripts\stop-live.ps1"
