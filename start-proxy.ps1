<#
    start-proxy.ps1 -- launch the AgentRouter compatibility proxy.

    Every path is derived from this script's own location, so the repository
    can be cloned anywhere by any user.

    Does five things, in order:
      0. Refuses to start a second instance on an already-healthy port.
      1. Loads .env if one is present (values are never printed).
      2. Activates .venv if one is present, else falls back to system Python.
      3. Verifies fastapi / uvicorn / httpx are importable.
      4. Verifies a credential exists WITHOUT printing it.
      5. Starts uvicorn bound to 127.0.0.1 (loopback only).

    Usage:
      .\start-proxy.ps1                  # interactive, Ctrl+C to stop
      .\start-proxy.ps1 -Port 8790       # different port
      .\start-proxy.ps1 -Verbose_        # extra diagnostics
      .\start-proxy.ps1 -Service         # supervised, rotating log file
#>

[CmdletBinding()]
param(
    [int]$Port = 8787,
    [switch]$Verbose_,
    # Set by the scheduled task: log to a rotating file instead of the console.
    [switch]$Service,
    # Hard ceiling per log file. One previous generation is kept as .1, so
    # total on-disk log usage is bounded at roughly 2 x this value.
    [int]$MaxLogKB = 1024
)

$ErrorActionPreference = 'Stop'

# Script-directory anchored. Works from any clone location and any username.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# ---------------------------------------------------------------------------
# Logging. Console when run interactively, rotating file under -Service.
#
# Nothing written here can contain credentials, prompts, or model output: the
# credential check below prints a length only, and the proxy's own log() emits
# structural diagnostics (event types, frame counts, drop reasons) exclusively.
# ---------------------------------------------------------------------------
$script:LogPath   = $null
$script:MaxBytes  = $MaxLogKB * 1KB
$script:LineCount = 0

if ($Service) {
    $logDir = Join-Path $root 'logs'
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    $script:LogPath = Join-Path $logDir 'proxy.log'
}

function Invoke-LogRotation {
    if (-not $script:LogPath) { return }
    if (-not (Test-Path $script:LogPath)) { return }
    if ((Get-Item $script:LogPath).Length -le $script:MaxBytes) { return }
    $old = "$($script:LogPath).1"
    if (Test-Path $old) { Remove-Item $old -Force -ErrorAction SilentlyContinue }
    Move-Item $script:LogPath $old -Force -ErrorAction SilentlyContinue
}

function Write-Line {
    param([string]$Text, [string]$Color = 'Gray')
    if (-not $script:LogPath) { Write-Host $Text -ForegroundColor $Color; return }
    # Size is checked every 50 lines rather than every line so a busy proxy
    # does not pay a stat() syscall per log record.
    if (($script:LineCount % 50) -eq 0) { Invoke-LogRotation }
    $script:LineCount++
    $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    try { Add-Content -LiteralPath $script:LogPath -Value "$stamp $Text" -Encoding utf8 -ErrorAction Stop } catch { }
}

function Write-Step { param([string]$m) Write-Line "[start] $m" 'Cyan' }
function Write-Ok   { param([string]$m) Write-Line "[ ok  ] $m" 'Green' }
function Write-Warn { param([string]$m) Write-Line "[warn ] $m" 'Yellow' }
function Write-Err  { param([string]$m) Write-Line "[fail ] $m" 'Red' }

# ---------------------------------------------------------------------------
# 0. Fast single-instance guard.
#
# Runs before any Python is spawned so that re-invocation against a healthy
# proxy costs one loopback GET and nothing else. Exits 0 so Task Scheduler
# records success and its restart-on-failure policy stays out of the way.
# ---------------------------------------------------------------------------
function Test-OurProxy {
    param([int]$OnPort)
    try {
        $probe = Invoke-RestMethod "http://127.0.0.1:$OnPort/_health" -TimeoutSec 5 -ErrorAction Stop
        return ($probe.status -eq 'ok' -and $probe.listen -eq "127.0.0.1:$OnPort")
    } catch { return $false }
}

function Get-PortOwners {
    param([int]$OnPort)
    $conns = $null
    try { $conns = Get-NetTCPConnection -LocalPort $OnPort -State Listen -ErrorAction SilentlyContinue } catch { }
    if (-not $conns) { return @() }
    return @($conns | Select-Object -ExpandProperty OwningProcess -Unique)
}

function Show-PortConflict {
    param([int]$OnPort, [int[]]$OwnerPids)
    $names = @()
    foreach ($procId in $OwnerPids) {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc) { $names += "$($proc.ProcessName) (PID $procId)" } else { $names += "PID $procId" }
    }
    Write-Err "Port $OnPort is occupied by an unrelated process: $($names -join ', ')"
    Write-Err 'It did not answer /_health as this proxy. Leaving it running and untouched.'
    Write-Err "Free the port yourself, or start the proxy elsewhere with -Port <number>."
}

$owners = Get-PortOwners -OnPort $Port
if ($owners.Count -gt 0) {
    if (Test-OurProxy -OnPort $Port) {
        Write-Ok "proxy already healthy on 127.0.0.1:$Port (PID $($owners -join ', ')) -- nothing to do"
        exit 0
    }
    # Someone else's process. Report it precisely and leave it alone; killing an
    # unidentified process on the user's machine is never this script's call.
    Show-PortConflict -OnPort $Port -OwnerPids $owners
    exit 2
}

# ---------------------------------------------------------------------------
# 1. Optional .env (never required, never printed)
#
# Only KEY=VALUE lines are read, and only into this process. Nothing is echoed,
# so a credential placed here cannot reach the console or the log file.
# ---------------------------------------------------------------------------
$envFile = Join-Path $root '.env'
if (Test-Path $envFile) {
    $loaded = 0
    foreach ($line in (Get-Content -LiteralPath $envFile -ErrorAction SilentlyContinue)) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        $split = $trimmed.IndexOf('=')
        if ($split -lt 1) { continue }
        $name  = $trimmed.Substring(0, $split).Trim()
        $value = $trimmed.Substring($split + 1).Trim().Trim('"').Trim("'")
        if (-not $name -or -not $value) { continue }
        Set-Item -Path "Env:$name" -Value $value
        $loaded++
    }
    Write-Ok ".env loaded ($loaded variable(s), values not displayed)"
}

# ---------------------------------------------------------------------------
# 2. Locate Python
# ---------------------------------------------------------------------------
Write-Step 'Locating Python interpreter'

$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$activate   = Join-Path $root '.venv\Scripts\Activate.ps1'

if (Test-Path $venvPython) {
    if (Test-Path $activate) {
        try { . $activate; Write-Ok 'venv activated' }
        catch { Write-Warn 'venv activation script failed; using its python.exe directly' }
    }
    $python = $venvPython
} else {
    Write-Warn 'no .venv found -- falling back to system Python'
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $python) { $python = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $python) {
        Write-Err 'No Python interpreter found on PATH. Install Python 3.10+ and re-run.'
        exit 1
    }
}

$pyVersion = & $python --version 2>&1
Write-Ok "interpreter: $pyVersion"

# ---------------------------------------------------------------------------
# 3. Verify dependencies
# ---------------------------------------------------------------------------
Write-Step 'Verifying dependencies (fastapi, uvicorn, httpx)'

$checkScript = @'
import importlib, sys
missing = []
for mod in ("fastapi", "uvicorn", "httpx"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<8} {getattr(m, '__version__', 'unknown')}")
    except ImportError:
        missing.append(mod)
if missing:
    print("MISSING:" + ",".join(missing))
    sys.exit(3)
'@

# Native stderr is piped into the log on purpose from here on. Under the
# script-wide 'Stop' preference the first stderr line of a native command is
# wrapped in an ErrorRecord and raised as a terminating NativeCommandError,
# which would abort the script -- pip in particular writes notices there. The
# preference is restored once native output stops being consumed.
$ErrorActionPreference = 'Continue'

# Fed over stdin (python -). Passing a multi-line script via -c lets PowerShell's
# native-argument handling strip the embedded quotes and corrupt the source.
$depOutput = $checkScript | & $python - 2>&1
$depExit = $LASTEXITCODE
$depOutput | ForEach-Object { Write-Line ([string]$_) }

if ($depExit -ne 0) {
    Write-Warn 'Dependencies missing -- installing from requirements.txt'
    & $python -m pip install -r (Join-Path $root 'requirements.txt') 2>&1 |
        ForEach-Object { Write-Line ([string]$_) }
    if ($LASTEXITCODE -ne 0) {
        Write-Err 'pip install failed. Install manually: python -m pip install -r requirements.txt'
        exit 1
    }
    $checkScript | & $python - | ForEach-Object { Write-Line ([string]$_) }
    if ($LASTEXITCODE -ne 0) { Write-Err 'Dependencies still unavailable after install.'; exit 1 }
}
Write-Ok 'all dependencies present'

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# 4. Check for a credential -- never print its value
#
# A credential here is entirely optional. Claude Code sends its own token on
# every request and the proxy forwards it untouched; the environment is only a
# fallback for clients that send nothing.
# ---------------------------------------------------------------------------
Write-Step 'Checking for a credential in the environment'

$keyVar = $null
$keyLen = 0
foreach ($name in @('ANTHROPIC_AUTH_TOKEN', 'AGENTROUTER_API_KEY', 'ANTHROPIC_API_KEY')) {
    $val = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($val)) {
        $val = [Environment]::GetEnvironmentVariable($name, 'User')
        if (-not [string]::IsNullOrWhiteSpace($val)) {
            # Make the user-scoped value visible to this process and to uvicorn.
            Set-Item -Path "Env:$name" -Value $val
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($val)) { $keyVar = $name; $keyLen = $val.Length; break }
}

if ($keyVar) {
    # Length only. The key itself is never written to the console or to a log.
    Write-Ok "credential found in `$env:$keyVar (length $keyLen, value not displayed)"
} else {
    Write-Warn 'No ANTHROPIC_AUTH_TOKEN / AGENTROUTER_API_KEY / ANTHROPIC_API_KEY set.'
    Write-Warn 'That is fine if Claude Code sends its own credential per request'
    Write-Warn '(it does when ANTHROPIC_AUTH_TOKEN is set in ~/.claude/settings.json).'
    Write-Warn 'The proxy forwards whatever credential the client presents.'
}

# ---------------------------------------------------------------------------
# 5. Start uvicorn on loopback only
# ---------------------------------------------------------------------------
if ($Verbose_) { $env:PROXY_VERBOSE = '1' }

# The proxy's diagnostics can carry U+FFFD: an unusable upstream body is decoded
# with errors="replace" before a short preview of it is logged. On a console
# whose codepage is not UTF-8 -- cp1251 here -- print() then raises
# UnicodeEncodeError from inside the request handler, which turns a reported
# upstream failure into a 500 and a traceback on stderr. UTF-8 with replacement
# on the child's own streams keeps a diagnostic from ever breaking a request.
$env:PYTHONIOENCODING = 'utf-8:replace'

# And decode that child's output as UTF-8 on this side, so the bytes reaching the
# log file are the ones Python wrote. Absent a console host this is unavailable,
# which costs nothing but fidelity on non-ASCII log lines.
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }

# So /_health reports the address it is genuinely bound to, which is what the
# single-instance guard above compares against.
$env:PROXY_PORT = "$Port"

$upstream = $env:AGENTROUTER_UPSTREAM
if ([string]::IsNullOrWhiteSpace($upstream)) { $upstream = 'https://agentrouter.org' }

Write-Line ''
Write-Line '  AgentRouter compatibility proxy' 'White'
Write-Line "  upstream : $upstream"  'DarkGray'
Write-Line "  listen   : http://127.0.0.1:$Port (loopback only)" 'DarkGray'
Write-Line "  stats    : http://127.0.0.1:$Port/_stats" 'DarkGray'
Write-Line "  health   : http://127.0.0.1:$Port/_health" 'DarkGray'
if ($Service) { Write-Line "  log      : $($script:LogPath) (rotates at $MaxLogKB KB)" 'DarkGray' }
else          { Write-Line '  stop     : Ctrl+C' 'DarkGray' }
Write-Line ''

# A listener left behind by a supervisor that is no longer watching it is worse
# than no listener at all: /_health still answers, so every later start decides
# the proxy is fine and exits, and the restart that was meant to pick up new code
# silently does nothing. Only direct children of this process are stopped, so an
# unrelated python on the machine is never touched.
function Stop-OwnUvicorn {
    $childName = Split-Path -Leaf $python
    $kids = @()
    try {
        $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $PID" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq $childName })
    } catch { }
    foreach ($kid in $kids) {
        Write-Warn "stopping leftover uvicorn (PID $($kid.ProcessId))"
        Stop-Process -Id $kid.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

# Merge uvicorn's stderr into the rotating log. Its output is startup banners and
# the proxy's own [proxy] structural diagnostics -- no credentials, prompts, or
# model output are ever emitted there.
#
# The preference is relaxed around the call because `2>&1` wraps every stderr line
# of a native command in an ErrorRecord: under 'Stop' the first one -- a uvicorn
# warning about a malformed request, say -- is a terminating NativeCommandError
# that kills this supervisor with exit code 1 and takes its uvicorn child with it,
# leaving no explanation in the log because the failure never reaches Write-Line.
# stderr is meant to be logged here, not to be fatal. Whatever ends the run, the
# python this function started is stopped before returning, so no orphan is left
# holding the port.
function Invoke-Uvicorn {
    param([int]$OnPort)
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    # Reported through a script variable rather than a return value: a function
    # whose try block returns and whose finally block still has work to do is one
    # more thing to be wrong about, and the exit code is the one fact the loop
    # below has to get right.
    $script:LastUvicornExit = $null
    try {
        & $python -m uvicorn proxy:app --host 127.0.0.1 --port $OnPort --log-level warning --no-access-log 2>&1 |
            ForEach-Object { Write-Line ([string]$_) }
        $script:LastUvicornExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
        Stop-OwnUvicorn
    }
}

# --host 127.0.0.1 is what keeps this loopback-only. Never change it to 0.0.0.0:
# the proxy forwards whatever credential the client presents, so a LAN-reachable
# listener lets any host on the network spend your API quota.
if ($Service) {
    # Supervisor loop. Task Scheduler's RestartOnFailure only fires when the
    # task itself ends in failure, and a killed uvicorn lets this wrapper exit
    # cleanly -- so the scheduler would not react. Restarting here recovers in
    # seconds instead of waiting for the watchdog repetition.
    $backoff       = 5
    $maxBackoff    = 60
    $rapidFailures = 0

    while ($true) {
        $startedAt = Get-Date

        try {
            Invoke-Uvicorn -OnPort $Port
        } catch {
            # Downgrading stderr inside Invoke-Uvicorn removes the one error that
            # used to reach here, but a supervisor that can die without saying why
            # is exactly the failure this loop exists to prevent. Anything else
            # gets logged and treated as a crashed run rather than ending the
            # script.
            Write-Err "supervisor caught: $($_.Exception.Message)"
        }
        $code    = if ($null -ne $script:LastUvicornExit) { $script:LastUvicornExit } else { 'unknown' }
        $ranFor  = ((Get-Date) - $startedAt).TotalSeconds
        Write-Warn "uvicorn exited with code $code after $([math]::Round($ranFor))s"

        # Did something else claim the port while we were down? If so this is
        # no longer our port to take -- report and stop rather than fight it.
        $taken = Get-PortOwners -OnPort $Port
        if ($taken.Count -gt 0) {
            Write-Err "Port $Port was claimed by another process (PID $($taken -join ', ')) while restarting. Stopping."
            exit 2
        }

        # A run that lasted a while is evidence of health, so reset the backoff.
        # Runs that die immediately are counted; enough of them means something
        # is genuinely broken and looping cannot fix it.
        if ($ranFor -ge 60) {
            $backoff = 5
            $rapidFailures = 0
        } else {
            $rapidFailures++
        }

        if ($rapidFailures -ge 10) {
            Write-Err "uvicorn failed $rapidFailures times in rapid succession -- giving up."
            Write-Err 'Exiting non-zero so Task Scheduler can retry later instead of looping here.'
            exit 1
        }

        Write-Warn "restarting in ${backoff}s (consecutive rapid failures: $rapidFailures)"
        Start-Sleep -Seconds $backoff
        $backoff = [Math]::Min($backoff * 2, $maxBackoff)
        Write-Step "restarting uvicorn on 127.0.0.1:$Port"
    }
}

& $python -m uvicorn proxy:app --host 127.0.0.1 --port $Port --log-level warning --no-access-log
