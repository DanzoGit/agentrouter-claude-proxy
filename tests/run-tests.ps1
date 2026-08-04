<#
    run-tests.ps1 -- run the full compatibility suite against a mock upstream.

    Nothing here touches the real AgentRouter service, a real credential, or a
    proxy you may already have running on 8787. A mock upstream and a throwaway
    proxy instance are started on isolated ports and torn down afterwards.

    Usage:
      .\tests\run-tests.ps1
      .\tests\run-tests.ps1 -MockPort 9788 -ProxyPort 9789
#>

[CmdletBinding()]
param(
    [int]$MockPort  = 8788,
    [int]$ProxyPort = 8789
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here

function Test-PortFree {
    param([int]$P)
    try {
        $c = Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction SilentlyContinue
        return -not $c
    } catch { return $true }
}

foreach ($p in @($MockPort, $ProxyPort)) {
    if (-not (Test-PortFree -P $p)) {
        Write-Host "[fail ] port $p is already in use. Pick another with -MockPort / -ProxyPort." -ForegroundColor Red
        exit 2
    }
}

$venvPython = Join-Path $root '.venv\Scripts\python.exe'
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $python) { $python = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $python) {
        Write-Host '[fail ] No Python interpreter found. Install Python 3.10+ and re-run.' -ForegroundColor Red
        exit 1
    }
}

Write-Host "[test ] interpreter : $(& $python --version 2>&1)" -ForegroundColor Cyan
Write-Host "[test ] mock  upstream on 127.0.0.1:$MockPort" -ForegroundColor Cyan
Write-Host "[test ] proxy under test on 127.0.0.1:$ProxyPort" -ForegroundColor Cyan

$mockProc = $null
$proxyProc = $null
$exitCode = 1

try {
    # ---- 1. syntax check -------------------------------------------------
    Write-Host "`n[test ] compiling sources" -ForegroundColor Cyan
    & $python -m py_compile (Join-Path $root 'proxy.py') `
                            (Join-Path $here 'mock_upstream.py') `
                            (Join-Path $here 'scenario_tests.py') `
                            (Join-Path $here 'disconnect_test.py')
    if ($LASTEXITCODE -ne 0) { throw 'py_compile failed' }
    Write-Host '[ ok  ] all sources compile' -ForegroundColor Green

    # ---- 2. start the mock upstream --------------------------------------
    $mockProc = Start-Process -FilePath $python -PassThru -WindowStyle Hidden `
        -WorkingDirectory $here `
        -ArgumentList @('-m', 'uvicorn', 'mock_upstream:app',
                        '--host', '127.0.0.1', '--port', "$MockPort",
                        '--log-level', 'warning', '--no-access-log')

    # ---- 3. start the proxy pointed at the mock --------------------------
    $env:AGENTROUTER_UPSTREAM = "http://127.0.0.1:$MockPort"
    $env:PROXY_PORT           = "$ProxyPort"
    # Keep the suite quick: the timeout scenarios do not need 120s to prove.
    $env:PROXY_PRIME_TIMEOUT_S = '10'

    $proxyProc = Start-Process -FilePath $python -PassThru -WindowStyle Hidden `
        -WorkingDirectory $root `
        -ArgumentList @('-m', 'uvicorn', 'proxy:app',
                        '--host', '127.0.0.1', '--port', "$ProxyPort",
                        '--log-level', 'warning', '--no-access-log')

    # ---- 4. wait for both to answer --------------------------------------
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:$ProxyPort/_health" -TimeoutSec 3 -ErrorAction Stop
            Invoke-RestMethod "http://127.0.0.1:$MockPort/_mock_stats" -TimeoutSec 3 -ErrorAction Stop | Out-Null
            if ($h.status -eq 'ok') { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw "proxy or mock did not become ready on 127.0.0.1:$ProxyPort / :$MockPort" }

    $health = Invoke-RestMethod "http://127.0.0.1:$ProxyPort/_health" -TimeoutSec 5
    Write-Host "[ ok  ] /_health -> status=$($health.status) listen=$($health.listen) upstream=$($health.upstream)" -ForegroundColor Green

    if ($health.listen -ne "127.0.0.1:$ProxyPort") {
        throw "proxy reports listen=$($health.listen) -- expected loopback 127.0.0.1:$ProxyPort"
    }

    # ---- 5. confirm the listener is loopback-only ------------------------
    $bound = Get-NetTCPConnection -LocalPort $ProxyPort -State Listen -ErrorAction SilentlyContinue
    $addrs = @($bound | Select-Object -ExpandProperty LocalAddress -Unique)
    if ($addrs -contains '0.0.0.0' -or $addrs -contains '::') {
        throw "proxy is bound to $($addrs -join ', ') -- expected 127.0.0.1 only"
    }
    Write-Host "[ ok  ] bound to $($addrs -join ', ') only (not 0.0.0.0)" -ForegroundColor Green

    $env:TEST_PROXY_URL = "http://127.0.0.1:$ProxyPort"
    $env:TEST_MOCK_URL  = "http://127.0.0.1:$MockPort"

    # ---- 6. scenario suite -----------------------------------------------
    Write-Host "`n[test ] scenario suite" -ForegroundColor Cyan
    & $python (Join-Path $here 'scenario_tests.py')
    $scenarioExit = $LASTEXITCODE

    # ---- 7. client disconnect --------------------------------------------
    Write-Host "`n[test ] client disconnect handling" -ForegroundColor Cyan
    & $python (Join-Path $here 'disconnect_test.py')
    $disconnectExit = $LASTEXITCODE

    # ---- 8. counters ------------------------------------------------------
    Write-Host "`n[test ] /_stats" -ForegroundColor Cyan
    $st = Invoke-RestMethod "http://127.0.0.1:$ProxyPort/_stats" -TimeoutSec 10
    $st | ConvertTo-Json -Depth 3 | Write-Host

    if ($scenarioExit -eq 0 -and $disconnectExit -eq 0) {
        Write-Host "`n[ ok  ] ALL TESTS PASSED" -ForegroundColor Green
        $exitCode = 0
    } else {
        Write-Host "`n[fail ] scenario=$scenarioExit disconnect=$disconnectExit" -ForegroundColor Red
        $exitCode = 1
    }
}
catch {
    Write-Host "[fail ] $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
}
finally {
    # Only ever stops the two processes this script started itself.
    foreach ($p in @($proxyProc, $mockProc)) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch { }
        }
    }
    Remove-Item Env:AGENTROUTER_UPSTREAM, Env:PROXY_PORT, Env:PROXY_PRIME_TIMEOUT_S, `
                Env:TEST_PROXY_URL, Env:TEST_MOCK_URL -ErrorAction SilentlyContinue
}

exit $exitCode
