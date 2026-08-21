<#
    ui-check.ps1 -- render /_ui against a mock upstream and screenshot it.

    A panel change can be reviewed this way without restarting a proxy that is
    carrying live traffic, and without spending a real credential: the instance
    under test points at tests/mock_upstream.py on loopback, and the panel loads
    no remote assets, so nothing leaves the machine.

    Usage:
      .\tests\ui-check.ps1
      .\tests\ui-check.ps1 -KeepRunning        # leave it up to poke at by hand
#>

[CmdletBinding()]
param(
    [int]$MockPort    = 8896,
    [int]$ProxyPort   = 8897,
    [string]$OutDir   = (Join-Path $env:TEMP 'proxy-ui-check'),
    [switch]$KeepRunning
)

# Native tools write progress to stderr; under 'Stop' that alone kills the run.
$ErrorActionPreference = 'Continue'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here

function Write-Step { param([string]$m) Write-Host "[ui   ] $m" -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host "[ ok  ] $m" -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host "[warn ] $m" -ForegroundColor Yellow }
function Write-Err  { param([string]$m) Write-Host "[fail ] $m" -ForegroundColor Red }

foreach ($p in @($MockPort, $ProxyPort)) {
    if (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) {
        Write-Err "port $p is in use. Pick another with -MockPort / -ProxyPort."
        exit 2
    }
}

$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$python = if (Test-Path $venvPython) { $venvPython } else { (Get-Command python).Source }
if (-not $python) { Write-Err 'no Python interpreter found'; exit 1 }

$chrome = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$mockProc = $null
$proxyProc = $null

try {
    Write-Step "mock upstream on 127.0.0.1:$MockPort"
    $mockProc = Start-Process -FilePath $python -PassThru -WindowStyle Hidden `
        -WorkingDirectory $here `
        -ArgumentList @('-m', 'uvicorn', 'mock_upstream:app',
                        '--host', '127.0.0.1', '--port', "$MockPort",
                        '--log-level', 'warning', '--no-access-log')

    $env:AGENTROUTER_UPSTREAM  = "http://127.0.0.1:$MockPort"
    $env:PROXY_PORT            = "$ProxyPort"
    $env:PROXY_STREAM_MODE     = 'reliable'
    $env:PROXY_PRIME_TIMEOUT_S = '8'
    $env:PYTHONIOENCODING      = 'utf-8:replace'

    Write-Step "proxy under test on 127.0.0.1:$ProxyPort"
    $proxyProc = Start-Process -FilePath $python -PassThru -WindowStyle Hidden `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $OutDir 'proxy.out') `
        -RedirectStandardError  (Join-Path $OutDir 'proxy.err') `
        -ArgumentList @('-m', 'uvicorn', 'proxy:app',
                        '--host', '127.0.0.1', '--port', "$ProxyPort",
                        '--log-level', 'warning', '--no-access-log')

    $base = "http://127.0.0.1:$ProxyPort"
    $up = $false
    foreach ($i in 1..40) {
        try {
            Invoke-RestMethod "$base/_health" -TimeoutSec 2 | Out-Null
            $up = $true; break
        } catch { Start-Sleep -Milliseconds 400 }
    }
    if (-not $up) { throw "proxy did not answer on $base/_health" }
    Write-Ok 'proxy answering'

    # A spread of outcomes, so the panel has something of every kind to draw:
    # clean streams of different lengths, a replayed one, a hard failure, a few
    # upstream statuses. Paced apart so the trace shows a rhythm, not one spike.
    $plan = @(
        @{ s = 'valid';                  stream = $true;  wait = 1 },
        @{ s = 'slow_valid';             stream = $true;  wait = 1 },
        @{ s = 'large_valid';            stream = $true;  wait = 2 },
        @{ s = 'flaky';                  stream = $true;  wait = 1 },
        @{ s = 'nonstream_valid';        stream = $false; wait = 1 },
        @{ s = 'truncated_post';         stream = $true;  wait = 2 },
        @{ s = 'slow_valid';             stream = $true;  wait = 1 },
        @{ s = 'http_429_saturated';     stream = $true;  wait = 1 },
        @{ s = 'valid_tool';             stream = $true;  wait = 2 },
        @{ s = 'remote_break';           stream = $true;  wait = 1 },
        @{ s = 'http_400';               stream = $false; wait = 1 },
        @{ s = 'slow_valid';             stream = $true;  wait = 2 },
        @{ s = 'empty_sse';              stream = $true;  wait = 1 },
        @{ s = 'valid';                  stream = $true;  wait = 1 },
        @{ s = 'always_break';           stream = $true;  wait = 2 },
        @{ s = 'valid';                  stream = $true;  wait = 1 },
        @{ s = 'nonstream_flaky';        stream = $false; wait = 1 },
        @{ s = 'slow_valid';             stream = $true;  wait = 0 }
    )

    $headers = @{
        'x-api-key'         = 'ui-check-not-a-real-key'
        'anthropic-version' = '2023-06-01'
        'content-type'      = 'application/json'
        'user-agent'        = 'claude-cli/2.0.14 (external, cli)'
        'x-app'             = 'cli'
    }

    Write-Step "$($plan.Count) requests through the proxy"
    foreach ($step in $plan) {
        $body = @{
            model      = "scenario:$($step.s)"
            max_tokens = 256
            stream     = $step.stream
            messages   = @(@{ role = 'user'; content = 'ui check' })
        } | ConvertTo-Json -Depth 5 -Compress

        try {
            Invoke-WebRequest "$base/v1/messages" -Method Post -Headers $headers `
                -Body $body -TimeoutSec 40 -UseBasicParsing | Out-Null
        } catch {
            # Failure scenarios are the point of half this list.
        }
        if ($step.wait) { Start-Sleep -Seconds $step.wait }
    }

    $stats = Invoke-RestMethod "$base/_stats" -TimeoutSec 5
    $events = Invoke-RestMethod "$base/_events?limit=600" -TimeoutSec 5
    Write-Ok ("requests=$($stats.total_requests) clean=$($stats.successful_requests) " +
              "failed=$($stats.failed_requests) events=$($events.events.Count)")

    if (-not $chrome) {
        Write-Warn 'Chrome not found: skipping screenshots'
    } else {
        # 500 is the narrowest viewport headless Chrome will report: ask for less
        # and it still lays out at 500, then crops the shot, which reads as a
        # broken layout when nothing is broken.
        # Dark is the panel default; ?theme=light asks for the other one, since
        # the theme no longer follows the system preference.
        $shots = @(
            @{ name = 'panel-dark.png';   size = '1440,1560'; query = '' },
            @{ name = 'panel-light.png';  size = '1440,1560'; query = '?theme=light' },
            @{ name = 'panel-mid.png';    size = '820,1900';  query = '' },
            @{ name = 'panel-narrow.png'; size = '500,2100';  query = '' }
        )
        foreach ($shot in $shots) {
            $out = Join-Path $OutDir $shot.name
            Remove-Item $out -ErrorAction SilentlyContinue
            $chromeArgs = @('--headless=new', '--disable-gpu', '--hide-scrollbars',
                      '--no-first-run', '--no-default-browser-check',
                      "--window-size=$($shot.size)",
                      '--virtual-time-budget=9000',
                      "--screenshot=$out", "$base/_ui$($shot.query)")
            & $chrome @chromeArgs 2>&1 | Out-Null
            if (Test-Path $out) {
                Write-Ok "$($shot.name) $([math]::Round((Get-Item $out).Length / 1kb))kb"
            } else {
                Write-Err "$($shot.name) was not written"
            }
        }
    }

    Write-Host ''
    Write-Ok "output in $OutDir"
    if ($KeepRunning) {
        Write-Warn "left running: $base/_ui  (proxy pid $($proxyProc.Id), mock pid $($mockProc.Id))"
    }
}
catch {
    Write-Err $_.Exception.Message
    exit 1
}
finally {
    if (-not $KeepRunning) {
        foreach ($p in @($proxyProc, $mockProc)) {
            if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        }
    }
}
