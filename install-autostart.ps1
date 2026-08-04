<#
    install-autostart.ps1 -- register a Windows scheduled task that starts the
    proxy at logon.

    Everything is derived at runtime: the repository path comes from this
    script's own location and the account comes from the current session, so no
    username or absolute path is hardcoded.

    No credential is stored in the task. None is needed -- Claude Code sends its
    own token with every request and the proxy forwards it untouched.

    Usage:
      .\install-autostart.ps1
      .\install-autostart.ps1 -TaskName MyProxy -Port 8790
      .\install-autostart.ps1 -WhatIf_        # show the plan, change nothing

    Remove it again with .\uninstall-autostart.ps1
#>

[CmdletBinding()]
param(
    [string]$TaskName = 'AgentRouterProxy',
    [int]$Port = 8787,
    # Seconds to wait after logon before starting, so the network stack is up.
    [int]$DelaySeconds = 8,
    # Rotating log ceiling in KB, passed through to start-proxy.ps1.
    [int]$MaxLogKB = 1024,
    [switch]$WhatIf_
)

$ErrorActionPreference = 'Stop'

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root 'start-proxy.ps1'

if (-not (Test-Path $script)) {
    Write-Host "[fail ] start-proxy.ps1 not found next to this script ($root)" -ForegroundColor Red
    exit 1
}

# Current interactive user, resolved at runtime -- never a hardcoded account.
$account = "$env:USERDOMAIN\$env:USERNAME"

$powershell = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $powershell) { $powershell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" }

$arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -Service -Port {1} -MaxLogKB {2}' -f `
             $script, $Port, $MaxLogKB

Write-Host ''
Write-Host '  Scheduled task to be registered' -ForegroundColor White
Write-Host "  name      : $TaskName"        -ForegroundColor DarkGray
Write-Host "  account   : $account"          -ForegroundColor DarkGray
Write-Host "  trigger   : at logon, +${DelaySeconds}s delay" -ForegroundColor DarkGray
Write-Host "  action    : $powershell"       -ForegroundColor DarkGray
Write-Host "  arguments : $arguments"        -ForegroundColor DarkGray
Write-Host "  workdir   : $root"             -ForegroundColor DarkGray
Write-Host "  listen    : 127.0.0.1:$Port (loopback only)" -ForegroundColor DarkGray
Write-Host '  credential: none stored in the task' -ForegroundColor DarkGray
Write-Host ''

if ($WhatIf_) {
    Write-Host '[ ok  ] -WhatIf_ given: nothing was changed.' -ForegroundColor Yellow
    exit 0
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[warn ] a task named '$TaskName' already exists -- it will be replaced." -ForegroundColor Yellow
}

$action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $account
$trigger.Delay = "PT${DelaySeconds}S"

# RunLevel Limited: no elevation. A loopback listener does not need admin, and
# the proxy must run as the same user whose Claude Code config it serves.
$principal = New-ScheduledTaskPrincipal -UserId $account -LogonType InteractiveToken -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

# Hidden: the proxy logs to a file under -Service, so there is nothing to watch.
$settings.Hidden = $true

try {
    Register-ScheduledTask -TaskName $TaskName `
        -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
        -Description 'Local Anthropic-compatible proxy for AgentRouter (loopback only, no credentials stored).' `
        -Force | Out-Null
} catch {
    Write-Host "[fail ] Register-ScheduledTask failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "[ ok  ] registered scheduled task '$TaskName'" -ForegroundColor Green

# Start it now so the proxy is up without waiting for the next logon. The
# script's own single-instance guard makes this safe if one is already running.
try {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host '[ ok  ] task started' -ForegroundColor Green
} catch {
    Write-Host "[warn ] could not start the task now: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host '[warn ] it will still run at your next logon.' -ForegroundColor Yellow
}

$up = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$Port/_health" -TimeoutSec 3 -ErrorAction Stop
        if ($h.status -eq 'ok') { $up = $true; break }
    } catch { }
}

Write-Host ''
if ($up) {
    Write-Host "[ ok  ] proxy is answering on http://127.0.0.1:$Port/_health" -ForegroundColor Green
} else {
    Write-Host "[warn ] proxy is not answering /_health yet. Check logs\proxy.log." -ForegroundColor Yellow
}

Write-Host ''
Write-Host '  Manage it with:' -ForegroundColor White
Write-Host "    Get-ScheduledTask   -TaskName $TaskName"  -ForegroundColor DarkGray
Write-Host "    Stop-ScheduledTask  -TaskName $TaskName   # stop this run"  -ForegroundColor DarkGray
Write-Host "    Disable-ScheduledTask -TaskName $TaskName # keep it down"  -ForegroundColor DarkGray
Write-Host "    .\uninstall-autostart.ps1                 # remove entirely" -ForegroundColor DarkGray
Write-Host ''
