<#
    uninstall-autostart.ps1 -- stop and remove the scheduled task created by
    install-autostart.ps1.

    Only ever touches the named task and, optionally, the uvicorn process that
    task started. Nothing else on the machine is stopped or deleted.

    Usage:
      .\uninstall-autostart.ps1
      .\uninstall-autostart.ps1 -TaskName MyProxy -Port 8790
      .\uninstall-autostart.ps1 -KeepRunning     # unregister but leave it up
#>

[CmdletBinding()]
param(
    [string]$TaskName = 'AgentRouterProxy',
    [int]$Port = 8787,
    [switch]$KeepRunning
)

$ErrorActionPreference = 'Stop'

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "[ ok  ] no scheduled task named '$TaskName' -- nothing to remove." -ForegroundColor Green
} else {
    try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[ ok  ] removed scheduled task '$TaskName'" -ForegroundColor Green
}

if ($KeepRunning) {
    Write-Host '[ ok  ] -KeepRunning given: the running proxy was left alone.' -ForegroundColor Green
    exit 0
}

# Stop the listener only after confirming it is this proxy. An unrelated process
# on the same port is reported and left untouched.
$conns = $null
try { $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue } catch { }

if (-not $conns) {
    Write-Host "[ ok  ] nothing listening on 127.0.0.1:$Port" -ForegroundColor Green
    exit 0
}

$isOurs = $false
try {
    $h = Invoke-RestMethod "http://127.0.0.1:$Port/_health" -TimeoutSec 5 -ErrorAction Stop
    if ($h.status -eq 'ok' -and $h.listen -eq "127.0.0.1:$Port") { $isOurs = $true }
} catch { }

$ownerPids = @($conns | Select-Object -ExpandProperty OwningProcess -Unique)

if (-not $isOurs) {
    Write-Host "[warn ] port $Port is held by a process that did not answer /_health as this proxy (PID $($ownerPids -join ', '))." -ForegroundColor Yellow
    Write-Host '[warn ] leaving it running and untouched.' -ForegroundColor Yellow
    exit 0
}

foreach ($procId in $ownerPids) {
    try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        Write-Host "[ ok  ] stopped proxy process PID $procId" -ForegroundColor Green
    } catch {
        Write-Host "[warn ] could not stop PID ${procId}: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
